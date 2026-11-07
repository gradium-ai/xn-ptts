// Lays the checkpoint out for the GPU from its q8 GGUF: one blob of u32 words, built once on
// the host, every tensor starting on a 16-byte boundary because the kernels read 16 bytes at a
// time. Offsets are returned in the units the kernels' uniforms use (see kernels.js):
//
//   int8 matrix   `[N][K]` as packed i8x4 words, then its f16 scales (one per 32 weights, the
//                 GGUF's q8_0 groups, split out of their blocks) as packed pairs: `wo` in
//                 16-byte words, `so` in words
//   f16 matrix    `[N][K]` as packed pairs (the GGUF's bf16, which f16 holds exactly at these
//                 magnitudes): `wo` in 16-byte words
//   f32 vector    as is: an offset in words

import { GGML, tensorBytes } from './gguf.js';

const f32buf = new Float32Array(1);
const u32buf = new Uint32Array(f32buf.buffer);

/** f32 -> f16 bits, round to nearest even. */
export function toHalf(v) {
  f32buf[0] = v;
  const x = u32buf[0];
  const sign = (x >>> 16) & 0x8000;
  const e = (x >>> 23) & 0xff;
  let man = x & 0x7fffff;
  if (e === 0xff) return sign | 0x7c00 | (man ? 0x200 : 0);
  const exp = e - 127 + 15;
  if (exp >= 31) return sign | 0x7c00;
  if (exp <= 0) {
    if (exp < -10) return sign;
    man = (man | 0x800000) >>> (1 - exp);
    return sign | ((man + 0xfff + ((man >>> 13) & 1)) >>> 13);
  }
  return sign | ((exp << 10) + ((man + 0xfff + ((man >>> 13) & 1)) >>> 13));
}

export function fromHalf(h) {
  const s = h & 0x8000 ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

/** A tensor as f32, whatever its stored type (q8_0 is dequantized). */
export function tensorF32(g, name) {
  const t = g.tensors.get(name);
  if (!t) throw new Error(`the checkpoint has no tensor ${name}`);
  const b = tensorBytes(g, name);
  const n = t.n;
  const aligned = b.byteOffset % 4 === 0;
  if (t.type === GGML.F32) return aligned ? new Float32Array(b.buffer, b.byteOffset, n).slice() : new Float32Array(b.slice().buffer);
  if (t.type === GGML.BF16) {
    const h = aligned ? new Uint16Array(b.buffer, b.byteOffset, n) : new Uint16Array(b.slice().buffer);
    const out = new Uint32Array(n);
    for (let i = 0; i < n; i++) out[i] = h[i] << 16;
    return new Float32Array(out.buffer);
  }
  if (t.type === GGML.F16) {
    const h = aligned ? new Uint16Array(b.buffer, b.byteOffset, n) : new Uint16Array(b.slice().buffer);
    return Float32Array.from(h, fromHalf);
  }
  if (t.type === GGML.Q8_0) {
    const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
    const out = new Float32Array(n);
    for (let blk = 0; blk < n / 32; blk++) {
      const d = fromHalf(dv.getUint16(blk * 34, true));
      for (let i = 0; i < 32; i++) out[blk * 32 + i] = d * dv.getInt8(blk * 34 + 2 + i);
    }
    return out;
  }
  throw new Error(`tensor ${name} has unsupported type ${t.type}`);
}

class Blob {
  constructor() { this.parts = []; this.words = 0; }
  push(u32) {
    if (this.words % 4) { this.parts.push(new Uint32Array(4 - (this.words % 4))); this.words += 4 - (this.words % 4); }
    const at = this.words;
    this.parts.push(u32);
    this.words += u32.length;
    return at;
  }
  finish() {
    const out = new Uint32Array(Math.ceil(this.words / 4) * 4);
    let o = 0;
    for (const p of this.parts) { out.set(p, o); o += p.length; }
    this.parts = [];
    return out;
  }
  f32(v) { return this.push(new Uint32Array(new Float32Array(v).buffer)); }
  f16(m) {
    const w = new Uint32Array(Math.ceil(m.length / 2));
    for (let i = 0; i < m.length; i += 2) w[i >> 1] = (toHalf(m[i]) | ((i + 1 < m.length ? toHalf(m[i + 1]) : 0) << 16)) >>> 0;
    return { wo: this.push(w) / 4 };
  }
  /** A q8_0 tensor's 34-byte blocks split into int8 words and f16 scale pairs. */
  q8(bytes) {
    const nb = bytes.length / 34;
    const ints = new Uint8Array(nb * 32);
    const scales = new Uint32Array(Math.ceil(nb / 2));
    for (let b = 0; b < nb; b++) {
      scales[b >> 1] |= (bytes[b * 34] | (bytes[b * 34 + 1] << 8)) << ((b & 1) * 16);
      ints.set(bytes.subarray(b * 34 + 2, b * 34 + 34), b * 32);
    }
    const wo = this.push(new Uint32Array(ints.buffer));
    return { wo: wo / 4, so: this.push(scales) };
  }
}

const cat = (...as) => { const o = new Float32Array(as.reduce((n, a) => n + a.length, 0)); let p = 0; for (const a of as) { o.set(a, p); p += a.length; } return o; };

/**
 * The flow net's time conditioning for its one LSD step (`s = 0`, `t = 1`): `(embed_0(s) +
 * embed_1(t)) / 2`, as `ptts`'s `SimpleMLPAdaLN::time_embedding`. It depends on nothing a frame
 * computes, so it is folded into the conditioning bias once.
 */
export function timeEmbedding(g, timeNorm) {
  const embed = (i, t) => {
    const p = `flow_lm.flow_net.time_embed.${i}`;
    const freqs = tensorF32(g, `${p}.freqs`);
    const w1 = tensorF32(g, `${p}.mlp.0.weight`), b1 = tensorF32(g, `${p}.mlp.0.bias`);
    const w2 = tensorF32(g, `${p}.mlp.2.weight`), b2 = tensorF32(g, `${p}.mlp.2.bias`);
    const alpha = tensorF32(g, `${p}.mlp.3.alpha`);
    const half = freqs.length;
    const e = new Float32Array(half * 2);
    for (let j = 0; j < half; j++) { e[j] = Math.cos(t * freqs[j]); e[half + j] = Math.sin(t * freqs[j]); }
    const D = b1.length;
    const h = new Float32Array(D);
    for (let n = 0; n < D; n++) { let a = b1[n]; for (let k = 0; k < e.length; k++) a += w1[n * e.length + k] * e[k]; h[n] = a / (1 + Math.exp(-a)); }
    const o = new Float32Array(D);
    for (let n = 0; n < D; n++) { let a = b2[n]; for (let k = 0; k < D; k++) a += w2[n * D + k] * h[k]; o[n] = a; }
    let norm;
    if (timeNorm === 'rms') norm = Math.sqrt(o.reduce((a, v) => a + v * v, 0) / D + 1e-5);
    else { const mean = o.reduce((a, v) => a + v, 0) / D; norm = Math.sqrt(o.reduce((a, v) => a + (v - mean) ** 2, 0) / (D - 1) + 1e-5); }
    return o.map((v, n) => (v / norm) * alpha[n]);
  };
  const a = embed(0, 0), b = embed(1, 1);
  return a.map((v, n) => (v + b[n]) / 2);
}

/** Builds the blob and the offsets for lm.js and codec.js. `g` is the parsed GGUF, `cfg` the
 *  checkpoint's config (already checked by support.js). */
export function buildWeights(g, cfg) {
  const fl = cfg.flow_lm;
  const b = new Blob();
  const T = (n) => tensorF32(g, n);
  const layers = [];
  for (let l = 0; l < fl.num_layers; l++) {
    const p = `flow_lm.transformer.layers.${l}`;
    layers.push({
      qkv: b.q8(tensorBytes(g, `${p}.self_attn.in_proj.weight`)),
      out: b.q8(tensorBytes(g, `${p}.self_attn.out_proj.weight`)),
      fc1: b.q8(tensorBytes(g, `${p}.linear1.weight`)),
      fc2: b.q8(tensorBytes(g, `${p}.linear2.weight`)),
      n1w: b.f32(T(`${p}.norm1.weight`)), n1b: b.f32(T(`${p}.norm1.bias`)),
      n2w: b.f32(T(`${p}.norm2.weight`)), n2b: b.f32(T(`${p}.norm2.bias`)),
    });
  }
  const fn = 'flow_lm.flow_net';
  const temb = timeEmbedding(g, fl.time_rms_norm ?? 'rms_var');
  const flow = {
    cond: { ...b.f16(T(`${fn}.cond_embed.weight`)), bo: b.f32(T(`${fn}.cond_embed.bias`).map((v, i) => v + temb[i])) },
    inproj: { ...b.f16(T(`${fn}.input_proj.weight`)), bo: b.f32(T(`${fn}.input_proj.bias`)) },
    blocks: [],
  };
  const adaW = [], adaB = [];
  for (let i = 0; i < fl.flow_depth; i++) {
    const p = `${fn}.res_blocks.${i}`;
    adaW.push(T(`${p}.adaLN_modulation.1.weight`)); adaB.push(T(`${p}.adaLN_modulation.1.bias`));
    flow.blocks.push({
      lw: b.f32(T(`${p}.in_ln.weight`)), lb: b.f32(T(`${p}.in_ln.bias`)),
      fc1: { ...b.f16(T(`${p}.mlp.0.weight`)), bo: b.f32(T(`${p}.mlp.0.bias`)) },
      fc2: { ...b.f16(T(`${p}.mlp.2.weight`)), bo: b.f32(T(`${p}.mlp.2.bias`)) },
    });
  }
  adaW.push(T(`${fn}.final_layer.adaLN_modulation.1.weight`)); adaB.push(T(`${fn}.final_layer.adaLN_modulation.1.bias`));
  // The six blocks' modulations and the final layer's in one [10240][512] GEMV.
  flow.ada = { ...b.f16(cat(...adaW)), bo: b.f32(cat(...adaB)), rows: adaB.reduce((n, a) => n + a.length, 0) };
  flow.final = { ...b.f16(T(`${fn}.final_layer.linear.weight`)), bo: b.f32(T(`${fn}.final_layer.linear.bias`)) };
  const top = {
    outNormW: b.f32(T('flow_lm.out_norm.weight')), outNormB: b.f32(T('flow_lm.out_norm.bias')),
    eosW: b.f32(T('flow_lm.out_eos.weight')), eosB: b.f32(T('flow_lm.out_eos.bias')),
    inLinear: b.f16(T('flow_lm.input_linear.weight')),
    // The conditioner sum, which the Rust side computes: the engine writes it here at load.
    cond: b.f32(new Float32Array(fl.d_model)),
  };
  const codec = buildCodec(g, b, cfg);
  const host = {
    bos: T('flow_lm.bos_emb'),
    inLinear: T('flow_lm.input_linear.weight'),
    // The text table, row = token id: a prompt's text rows are looked up on the host.
    embed: T('flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight'),
  };
  return { layers, flow, top, codec, host, blob: b };
}

/**
 * The Mimi decoder: convolutions as `[cout][k][cin]` (the GGUF's `[cout][cin][k]` reordered,
 * so output row m reads k * cin contiguous floats), each transposed convolution (stride s,
 * kernel 2s) as one `[cout][2 * cin]` matrix per output phase j, `[W[:, o, j + s] | W[:, o, j]]`,
 * applied to the rows `[x[t - 1], x[t]]`.
 */
function buildCodec(g, b, cfg) {
  const m = cfg.mimi;
  const T = (n) => tensorF32(g, n);
  const dims = (n) => {
    const t = g.tensors.get(n);
    if (!t) throw new Error(`the checkpoint has no tensor ${n}`);
    return t.dims;
  };
  const conv = (p) => {
    const [k, cin, cout] = dims(`${p}.weight`);
    const w = T(`${p}.weight`);
    const r = new Float32Array(w.length);
    for (let o = 0; o < cout; o++) for (let i = 0; i < cin; i++) for (let kk = 0; kk < k; kk++) r[(o * k + kk) * cin + i] = w[(o * cin + i) * k + kk];
    return { ...b.f16(r), bo: b.f32(T(`${p}.bias`)) };
  };
  const convT = (p, s) => {
    const [k, cout, cin] = dims(`${p}.weight`);
    if (k !== 2 * s) throw new Error(`${p}: kernel ${k} is not twice the stride ${s}`);
    const w = T(`${p}.weight`);
    const r = new Float32Array(s * cout * 2 * cin);
    for (let j = 0; j < s; j++) for (let o = 0; o < cout; o++) for (let c = 0; c < cin; c++) {
      const at = (j * cout + o) * 2 * cin;
      r[at + c] = w[(c * cout + o) * k + j + s];
      r[at + cin + c] = w[(c * cout + o) * k + j];
    }
    return { ...b.f16(r), wph: (cout * 2 * cin) / 8, bo: b.f32(T(`${p}.bias`)) };
  };
  const layers = [];
  for (let l = 0; l < m.transformer_num_layers; l++) {
    const p = `mimi.decoder_transformer.transformer.layers.${l}`;
    layers.push({
      qkv: b.f16(T(`${p}.self_attn.in_proj.weight`)), out: b.f16(T(`${p}.self_attn.out_proj.weight`)),
      fc1: b.f16(T(`${p}.linear1.weight`)), fc2: b.f16(T(`${p}.linear2.weight`)),
      n1w: b.f32(T(`${p}.norm1.weight`)), n1b: b.f32(T(`${p}.norm1.bias`)),
      n2w: b.f32(T(`${p}.norm2.weight`)), n2b: b.f32(T(`${p}.norm2.bias`)),
      ls1: b.f32(T(`${p}.layer_scale_1.scale`)), ls2: b.f32(T(`${p}.layer_scale_2.scale`)),
    });
  }
  const d = 'mimi.decoder.model';
  return {
    quant: b.f16(T('mimi.quantizer.output_proj.weight')),
    up: b.f32(T('mimi.upsample.convtr.convtr.weight')),
    embStd: b.f32(T('flow_lm.emb_std')), embMean: b.f32(T('flow_lm.emb_mean')),
    layers,
    seanet: {
      init: conv(`${d}.0.conv`),
      up: [convT(`${d}.2.convtr`, 6), convT(`${d}.5.convtr`, 5), convT(`${d}.8.convtr`, 4)],
      res: [3, 6, 9].map((i) => [conv(`${d}.${i}.block.1.conv`), conv(`${d}.${i}.block.3.conv`)]),
      final: conv(`${d}.11.conv`),
    },
  };
}
