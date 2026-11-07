// The Mimi decoder on the GPU: latents to 24 kHz PCM in chunks of C frames (1, 2, 4 or 8),
// each chunk one fixed dispatch list. See engine.js for where it sits.
//
//   codecIn   the latent denormalization, the quantizer's projection and the depthwise upsample
//             to 200 Hz (16 rows per frame), into the transformer's residual rows. The
//             upsample's overlap needs only the previous frame's latent, which is still in the
//             latents buffer, so it keeps no state.
//   2 layers  of the windowed attention transformer with layer scale: LayerNorm, GEMMs, RoPE
//             and an f32 ring of keys and values at the absolute 200 Hz position.
//   SEANet    every convolution is a GEMM whose A rows are read straight from a stage buffer:
//             a stage holds `ctx` rows of left context then the chunk's rows, so output row m of
//             a k-tap convolution reads the k * cin contiguous floats from input row m. A
//             transposed convolution with stride s and kernel 2s is s GEMMs (one per output
//             phase) over the row pairs [t - 1, t]. ELU is applied as A is loaded, so the stages
//             hold pre-activation values, which the residual adds need.
//   shift     each stage's last `ctx` rows move to its front for the next chunk.
//
// The chunk's first frame is `st.cpos`, which the host writes before the submit. Every kernel
// but the shift returns at once when that frame is past the end of speech.

import * as K from './kernels.js';
import { encode } from './dispatch.js';

export const CHUNKS = [1, 2, 4, 8];
const C_MAX = 8;
/** Samples per 80 ms frame. */
export const FRAME = 1920;
const D = 512;
const RING = K.CODEC_RING;

export class GpuCodec {
  /** @param {import('./lm.js').GpuLM} lm the LM, whose device, weights, state and latents it shares */
  constructor(lm, cw, { subgroups }) {
    this.lm = lm;
    this.cw = cw;
    this.subgroups = subgroups;
    const { device } = lm;
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const T = 16 * C_MAX;
    const stage = (name, ctx, rate, ch) => ({ name, ctx, rate, ch, buf: device.createBuffer({ size: (ctx + rate * T) * ch * 4, usage, label: name }) });
    // The stages that keep left context, in the shift kernel's binding order, then the rest.
    this.tr = stage('tr', 6, 1, 512);
    this.s0 = stage('s0', 1, 1, 512);
    this.s1 = stage('s1', 2, 6, 256);
    this.s1r = stage('s1r', 1, 6, 256);
    this.s2 = stage('s2', 2, 30, 128);
    this.s2r = stage('s2r', 1, 30, 128);
    this.s3 = stage('s3', 2, 120, 64);
    this.s3r = stage('s3r', 2, 120, 64);
    this.ctxStages = [this.tr, this.s0, this.s1, this.s1r, this.s2, this.s2r, this.s3, this.s3r];
    this.s1h = stage('s1h', 0, 6, 128);
    this.s2h = stage('s2h', 0, 30, 64);
    this.s3h = stage('s3h', 0, 120, 32);
    const f = (n, label) => device.createBuffer({ size: n * 4, usage, label });
    this.H = f(T * D, 'codec-h'); this.QKV = f(T * 3 * D, 'codec-qkv'); this.Q = f(T * D, 'codec-q');
    this.ATT = f(T * D, 'codec-att'); this.FF = f(T * 2048, 'codec-ff');
    this.KR = f(cw.layers.length * RING * D, 'codec-kring'); this.VR = f(cw.layers.length * RING * D, 'codec-vring');
    this.audio = f(C_MAX * FRAME, 'audio');
  }

  /** Registers one list per chunk size with `b`. */
  addLists(b) {
    const { lm, cw } = this;
    const sg = this.subgroups;
    const L = {
      gemm: b.layout('gemm', ['ro', 'ro', 'rw', 'ro', 'ro']),
      ln: b.layout('ln', ['ro', 'ro', 'rw', 'ro']),
      cin: b.layout('cin', ['ro', 'ro', 'rw', 'ro']),
      rope: b.layout('rope', ['ro', 'rw', 'rw', 'rw', 'ro']),
      attn: b.layout('cattn', ['ro', 'ro', 'ro', 'rw', 'ro']),
      shift: b.layout('shift', this.ctxStages.map(() => 'rw')),
    };
    const { W, st } = lm;
    const dummy = lm.DR;
    const g = (o) => b.pipe(`codec-gemm-${JSON.stringify(o)}`, L.gemm, K.gemm({ gate: true, ...o }));
    const pLN = b.pipe('codec-ln', L.ln, K.layerNorm({ sg, gate: true }));
    const pIN = b.pipe('codec-in', L.cin, K.codecIn());
    const pROPE = b.pipe('codec-rope', L.rope, K.ringRope());
    const pATT = b.pipe('codec-attn', L.attn, K.ringAttention({ sg }));
    const tiles = (M, N) => [Math.ceil(N / 64), Math.ceil(M / 64)];
    for (const C of CHUNKS) {
      const T = 16 * C;
      const list = `codec${C}`;
      const trX = this.tr.ctx * D; // the residual rows inside the transformer's stage
      b.add(list, pIN, [Math.ceil(T * D / 256)], { M: T, wo: cw.quant.wo, lg: cw.embStd, lb: cw.embMean, bo: cw.up, yb: trX }, [lm.LATS, W, this.tr.buf, st]);
      cw.layers.forEach((Ly, l) => {
        const ko = l * RING * D;
        b.add(list, pLN, [T], { ab: trX, lg: Ly.n1w, lb: Ly.n1b }, [this.tr.buf, W, this.H, st]);
        b.add(list, g({ K: 512, N: 1536 }), tiles(T, 1536), { M: T, arow: 512, wo: Ly.qkv.wo }, [this.H, W, this.QKV, dummy, st]);
        b.add(list, pROPE, [T], { M: T, ko }, [this.QKV, this.Q, this.KR, this.VR, st]);
        b.add(list, pATT, [T, 8], { ko }, [this.Q, this.KR, this.VR, this.ATT, st]);
        b.add(list, g({ K: 512, N: 512, res: 'ls' }), tiles(T, 512), { M: T, arow: 512, wo: Ly.out.wo, yb: trX, lso: Ly.ls1 }, [this.ATT, W, this.tr.buf, dummy, st]);
        b.add(list, pLN, [T], { ab: trX, lg: Ly.n2w, lb: Ly.n2b }, [this.tr.buf, W, this.H, st]);
        b.add(list, g({ K: 512, N: 2048, act: 'gelu' }), tiles(T, 2048), { M: T, arow: 512, wo: Ly.fc1.wo }, [this.H, W, this.FF, dummy, st]);
        b.add(list, g({ K: 2048, N: 512, res: 'ls' }), tiles(T, 512), { M: T, arow: 2048, wo: Ly.fc2.wo, yb: trX, lso: Ly.ls2 }, [this.FF, W, this.tr.buf, dummy, st]);
      });
      const audio = { buf: this.audio, ctx: 0 };
      const conv = (src, dst, cv, { cin, cout, k, elu, res }) => {
        const M = (dst === audio ? src.rate : dst.rate) * T;
        b.add(list, g({ K: k * cin, N: cout, eluA: elu, bias: true, res: res ? 'add' : 'none' }), tiles(M, cout),
          { M, ab: (src.ctx - (k - 1)) * cin, arow: cin, wo: cv.wo, bo: cv.bo, yb: dst.ctx * cout, rb: res ? res.ctx * cout : 0 },
          [src.buf, W, dst.buf, res ? res.buf : dummy, st]);
      };
      const convT = (src, dst, cv, { cin, cout, s }) => {
        const M = src.rate * T;
        b.add(list, g({ K: 2 * cin, N: cout, phases: s, eluA: true, bias: true }), [...tiles(M, cout), s],
          { M, ab: (src.ctx - 1) * cin, arow: cin, wo: cv.wo, wph: cv.wph, bo: cv.bo, yb: dst.ctx * cout },
          [src.buf, W, dst.buf, dummy, st]);
      };
      const S = cw.seanet;
      conv(this.tr, this.s0, S.init, { cin: 512, cout: 512, k: 7, elu: false });
      convT(this.s0, this.s1, S.up[0], { cin: 512, cout: 256, s: 6 });
      conv(this.s1, this.s1h, S.res[0][0], { cin: 256, cout: 128, k: 3, elu: true });
      conv(this.s1h, this.s1r, S.res[0][1], { cin: 128, cout: 256, k: 1, elu: true, res: this.s1 });
      convT(this.s1r, this.s2, S.up[1], { cin: 256, cout: 128, s: 5 });
      conv(this.s2, this.s2h, S.res[1][0], { cin: 128, cout: 64, k: 3, elu: true });
      conv(this.s2h, this.s2r, S.res[1][1], { cin: 64, cout: 128, k: 1, elu: true, res: this.s2 });
      convT(this.s2r, this.s3, S.up[2], { cin: 128, cout: 64, s: 4 });
      conv(this.s3, this.s3h, S.res[2][0], { cin: 64, cout: 32, k: 3, elu: true });
      conv(this.s3h, this.s3r, S.res[2][1], { cin: 32, cout: 64, k: 1, elu: true, res: this.s3 });
      conv(this.s3r, audio, S.final, { cin: 64, cout: 1, k: 3, elu: true });
      const pSH = b.pipe(`codec-shift${C}`, L.shift, K.shift(this.ctxStages, T));
      b.add(list, pSH, [Math.ceil(Math.max(...this.ctxStages.map((s) => s.ctx * s.ch)) / 256)], {}, this.ctxStages.map((s) => s.buf));
    }
  }

  setLists(lists) {
    this.lists = Object.fromEntries(CHUNKS.map((C) => [C, lists[`codec${C}`]]));
  }

  /** Before a text's first chunk: no left context (zeros, as the CPU's constant padding). */
  reset(enc) {
    for (const s of this.ctxStages) enc.clearBuffer(s.buf, 0, s.ctx * s.ch * 4);
  }

  /** Records the decode of a C-frame chunk (its first frame is st.cpos, written by the host). */
  encode(pass, C) { encode(pass, this.lists[C]); }

  dispatchesPerChunk(C) { return this.lists[C].length; }
}
