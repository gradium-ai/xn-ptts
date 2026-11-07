// The flow LM on the GPU: the decode step as one fixed dispatch list, and the prompt (voice or
// text rows) as another, which runs every row through each layer at once. See engine.js.

import * as K from './kernels.js';
import { encode } from './dispatch.js';

/** KV positions: the voice prompt, a chunk's text and its frames. */
export const S_MAX = 2048;
/** Frames one text chunk may make (82 s; `plan::frame_budget` gives at most 859). */
export const F_MAX = 1024;
/** Rows one prompt may have: a voice's frames or a chunk's tokens. */
export const ROWS_MAX = 512;
const D = 512;


const bits = (f) => new Uint32Array(new Float32Array([f]).buffer)[0];

export class GpuLM {
  /**
   * @param {GPUDevice} device
   * @param {object} w from `buildWeights`
   * @param {GPUBuffer} W the weight blob on the device
   * @param {{ subgroups: boolean, eosThreshold: number, layers: number }} o
   */
  constructor(device, w, W, o) {
    this.device = device;
    this.w = w;
    this.W = W;
    this.o = o;
    this.layers = o.layers;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const f = (n, label) => device.createBuffer({ size: Math.max(16, n * 4), usage, label });
    this.st = f(K.ST_WORDS, 'state');
    this.X = f(D, 'x'); this.QKV = f(3 * D, 'qkv'); this.FF = f(2048, 'ff');
    this.PART = f(8 * K.SPLITS * K.PART_STRIDE, 'part');
    this.TOUT = f(D, 'tout'); this.YA = f(D, 'ya'); this.ADA = f(w.flow.ada.rows, 'ada');
    this.CUR = f(32, 'cur'); this.HF = f(D, 'hf'); this.TF = f(D, 'tf');
    this.KC = f(this.layers * S_MAX * D / 2, 'kc'); this.VC = f(this.layers * S_MAX * D / 2, 'vc');
    this.LATS = f(F_MAX * 32, 'lats');
    this.ROWS = f(ROWS_MAX * D, 'prompt-rows'); this.PH = f(ROWS_MAX * D, 'prompt-h');
    this.PQKV = f(ROWS_MAX * 3 * D, 'prompt-qkv'); this.PQ = f(ROWS_MAX * D, 'prompt-q');
    this.PATT = f(ROWS_MAX * D, 'prompt-att'); this.PFF = f(ROWS_MAX * 2048, 'prompt-ff');
    this.DR = f(4096, 'dummy-ro'); this.DW = f(4096, 'dummy-rw');
  }

  /** Bytes of the f16 cache one layer holds. */
  static layerBytes() { return S_MAX * D * 2; }

  /** Registers the step and the prompt lists with `b`. */
  addLists(b) {
    const { w, W, st, X, QKV, FF, PART, TOUT, YA, ADA, CUR, HF, TF, KC, VC, DR, DW } = this;
    const sg = this.o.subgroups;
    const L = {
      gemv: b.layout('gemv', ['ro', 'ro', 'rw', 'ro', 'ro', 'rw', 'ro']),
      attn: b.layout('attn', ['ro', 'rw', 'rw', 'rw', 'ro']),
      inproj: b.layout('inproj', ['rw', 'ro', 'rw', 'ro']),
      finish: b.layout('finish', ['ro', 'ro', 'ro', 'rw', 'rw', 'rw']),
      ln: b.layout('ln', ['ro', 'ro', 'rw', 'ro']),
      gemm: b.layout('gemm', ['ro', 'ro', 'rw', 'ro', 'ro']),
      rope: b.layout('rope', ['ro', 'rw', 'rw', 'rw', 'ro']),
      pattn: b.layout('pattn', ['ro', 'ro', 'ro', 'rw', 'ro']),
      padv: b.layout('padv', ['rw']),
    };
    const geo = (N) => [K.gemvGroups(N)];
    const layerWords = S_MAX * D / 2;
    const gv = (key, o) => b.pipe(`lm-${key}`, L.gemv, K.gemv({ sg, ...o }));
    const gQKV = gv('qkv', { K: 512, N: 1536, wmode: 'q8', pro: 'ln', affine: true });
    const gOUT = gv('out', { K: 512, N: 512, wmode: 'q8', pro: 'combine', res: 'add' });
    const gFC1 = gv('fc1', { K: 512, N: 2048, wmode: 'q8', pro: 'ln', affine: true, actv: 'gelu' });
    const gFC2 = gv('fc2', { K: 2048, N: 512, wmode: 'q8', res: 'add' });
    const gCOND = gv('cond', { K: 512, N: 512, wmode: 'f16', pro: 'ln', affine: true, store: true, bias: true, actv: 'silu' });
    const gADA = gv('ada', { K: 512, N: w.flow.ada.rows, wmode: 'f16', bias: true });
    const gB1 = gv('b1', { K: 512, N: 512, wmode: 'f16', pro: 'ln', affine: true, modulate: true, eps: '1e-6', bias: true, actv: 'silu' });
    const gB2 = gv('b2', { K: 512, N: 512, wmode: 'f16', bias: true, res: 'gate' });
    // One LSD step from the noise: the latent is noise + flow (`res: add` onto `cur`).
    const gFIN = gv('fin', { K: 512, N: 32, wmode: 'f16', pro: 'ln', modulate: true, eps: '1e-6', bias: true, res: 'add' });
    const aATT = b.pipe('lm-attn', L.attn, K.attention({ sg }));
    const iIN = b.pipe('lm-in', L.inproj, K.inproj());
    const fFIN = b.pipe('lm-finish', L.finish, K.finish({ sg }));

    // gemv binds: x, wq, y, md, part, aux, st
    const bind = (x, y, { md = DR, part = DR, aux = DW } = {}) => [x, W, y, md, part, aux, st];
    w.layers.forEach((Ly, l) => {
      b.add('step', gQKV, geo(1536), { wo: Ly.qkv.wo, so: Ly.qkv.so, lg: Ly.n1w, lb: Ly.n1b }, bind(X, QKV));
      b.add('step', aATT, [K.SPLITS, 8], { ko: l * layerWords }, [QKV, KC, VC, PART, st]);
      b.add('step', gOUT, geo(512), { wo: Ly.out.wo, so: Ly.out.so }, bind(DR, X, { part: PART }));
      b.add('step', gFC1, geo(2048), { wo: Ly.fc1.wo, so: Ly.fc1.so, lg: Ly.n2w, lb: Ly.n2b }, bind(X, FF));
      b.add('step', gFC2, geo(512), { wo: Ly.fc2.wo, so: Ly.fc2.so }, bind(FF, X));
    });
    b.add('step', gCOND, geo(512), { wo: w.flow.cond.wo, bo: w.flow.cond.bo, lg: w.top.outNormW, lb: w.top.outNormB }, bind(X, YA, { aux: TOUT }));
    b.add('step', gADA, geo(w.flow.ada.rows), { wo: w.flow.ada.wo, bo: w.flow.ada.bo }, bind(YA, ADA));
    b.add('step', iIN, [2], { wo: w.flow.inproj.wo, bo: w.flow.inproj.bo }, [CUR, W, HF, st]);
    w.flow.blocks.forEach((B, i) => {
      b.add('step', gB1, geo(512), { wo: B.fc1.wo, bo: B.fc1.bo, lg: B.lw, lb: B.lb, sh: i * 1536, sc: i * 1536 + 512 }, bind(HF, TF, { md: ADA }));
      b.add('step', gB2, geo(512), { wo: B.fc2.wo, bo: B.fc2.bo, ga: i * 1536 + 1024 }, bind(TF, HF, { md: ADA }));
    });
    const fb = w.flow.blocks.length * 1536;
    b.add('step', gFIN, geo(32), { wo: w.flow.final.wo, bo: w.flow.final.bo, sh: fb, sc: fb + 512 }, bind(HF, CUR, { md: ADA }));
    b.add('step', fFIN, [1], { wo: w.top.inLinear.wo, lg: w.top.eosW, lb: w.top.eosB, sh: w.top.cond, a0: bits(this.o.eosThreshold) }, [CUR, TOUT, W, X, this.LATS, st]);

    // The prompt: every row through each layer at once (st.rows rows from position st.pos).
    const { ROWS, PH, PQKV, PQ, PATT, PFF } = this;
    const pLN = b.pipe('lm-pln', L.ln, K.layerNorm({ sg }));
    const mm = (key, o) => b.pipe(`lm-p${key}`, L.gemm, K.gemm({ wmode: 'q8', rowsFromState: true, ...o }));
    const pQKV = mm('qkv', { K: 512, N: 1536 });
    const pOUT = mm('out', { K: 512, N: 512, res: 'addy' });
    const pFC1 = mm('fc1', { K: 512, N: 2048, act: 'gelu' });
    const pFC2 = mm('fc2', { K: 2048, N: 512, res: 'addy' });
    const pROPE = b.pipe('lm-prope', L.rope, K.promptRope());
    const pATT = b.pipe('lm-pattn', L.pattn, K.promptAttention({ sg }));
    const pADV = b.pipe('lm-padv', L.padv, K.promptAdvance());
    const tiles = (N) => (M) => [Math.ceil(N / 64), Math.ceil(M / 64)];
    const rows = (M) => [M];
    w.layers.forEach((Ly, l) => {
      b.add('prompt', pLN, rows, { lg: Ly.n1w, lb: Ly.n1b }, [ROWS, W, PH, st]);
      b.add('prompt', pQKV, tiles(1536), { arow: 512, wo: Ly.qkv.wo, so: Ly.qkv.so }, [PH, W, PQKV, DR, st]);
      b.add('prompt', pROPE, rows, { ko: l * layerWords }, [PQKV, PQ, KC, VC, st]);
      b.add('prompt', pATT, (M) => [M, 8], { ko: l * layerWords }, [PQ, KC, VC, PATT, st]);
      b.add('prompt', pOUT, tiles(512), { arow: 512, wo: Ly.out.wo, so: Ly.out.so }, [PATT, W, ROWS, DR, st]);
      b.add('prompt', pLN, rows, { lg: Ly.n2w, lb: Ly.n2b }, [ROWS, W, PH, st]);
      b.add('prompt', pFC1, tiles(2048), { arow: 512, wo: Ly.fc1.wo, so: Ly.fc1.so }, [PH, W, PFF, DR, st]);
      b.add('prompt', pFC2, tiles(512), { arow: 2048, wo: Ly.fc2.wo, so: Ly.fc2.so }, [PFF, W, ROWS, DR, st]);
    });
    b.add('prompt', pADV, [1], {}, [st]);
  }

  setLists(lists) {
    this.stepList = lists.step;
    this.promptList = lists.prompt;
  }

  /** Records `n` decode steps. */
  encodeSteps(pass, n) { encode(pass, this.stepList, n); }

  /** Records the prompt: `rows` rows of `ROWS`, from position `st.pos`, which the host has
   *  written along with `st.rows = rows`. */
  encodePrompt(pass, rows) { encode(pass, this.promptList, 1, rows); }

  /** The state for a chunk (or a voice's prompt): everything the host sets. */
  stateWords({ pos, rows = 0, seed = 0, temperature = 0.3, after = 1, limit = F_MAX }) {
    const st = new Uint32Array(K.ST_WORDS);
    st[K.ST.pos] = pos; st[K.ST.rows] = rows; st[K.ST.seed] = seed >>> 0;
    st[K.ST.scale] = bits(Math.sqrt(temperature)); st[K.ST.after] = after; st[K.ST.limit] = limit;
    return st;
  }

  /** `input_linear(row) + conditions`: the first step's input, from the BOS row. */
  inputRow(row, cond) {
    const m = this.w.host.inLinear;
    const out = new Float32Array(D);
    for (let n = 0; n < D; n++) { let a = cond[n]; for (let k = 0; k < 32; k++) a += m[n * 32 + k] * row[k]; out[n] = a; }
    return out;
  }
}
