// Parts of this file are adapted from blixt/phonon (https://github.com/blixt/phonon),
// Copyright 2026 Blixt, licensed under the Apache License, Version 2.0, and were modified for
// this engine. This file is therefore under the Apache License 2.0 only (LICENSE-APACHE), not
// the MIT option of the rest of the repository. See NOTICE.
//
// WGSL for the WebGPU engine (see engine.js for how the pieces fit together).
//
// Every generator returns the shader for one configuration, so its loops and branches on
// constants fold away. `sg` picks how a kernel sums across threads: with subgroup operations,
// which assume 32-lane subgroups laid out along `local_invocation_index` (the engine only asks
// for them on adapters whose subgroups are exactly 32 wide), or with workgroup shared memory,
// which runs on any adapter.
//
// Weights are one blob of u32 words (weights.js): int8 matrices `[N][K]` as packed i8x4 words
// with one f16 scale per 32 weights, f16 matrices as packed pairs, f32 vectors as they are. A
// kernel takes its tensors' blob offsets from its uniform `P`. Every LM and codec kernel reads
// the state buffer `st` (ST below) and returns at once when its work is past the end of
// speech, so the steps and chunks a submit holds beyond the end cost little.

/** `st` slots (u32): the KV position the next step writes, the frame it makes, the frame count
 *  at which to stop (0: no end of speech seen yet), a stop flag the LM kernels check, the noise
 *  seed and scale (f32 bits), the frames kept after the end-of-speech frame, the frame limit,
 *  the first frame of the codec chunk being decoded, and the rows the prompt being run has. The
 *  host writes all but `frame`, `end` and `stop`; those the LM's last kernel advances. */
export const ST = { pos: 0, frame: 1, end: 2, stop: 3, seed: 4, scale: 5, after: 6, limit: 7, cpos: 8, rows: 9 };
export const ST_WORDS = 16;

/** The uniform's fields, all u32, in order (f32 values are stored as their bits). */
export const P_FIELDS = ['wo', 'so', 'bo', 'lg', 'lb', 'sh', 'sc', 'ga', 'xo', 'yo', 'ko', 'a0', 'M', 'ab', 'arow', 'wph', 'yb', 'rb', 'lso', 'p0'];
export const P_BYTES = 80;
const STRUCT_P = `struct P { ${P_FIELDS.map((f) => `${f}: u32`).join(', ')} }`;

/** Splits of the KV range one head's decode attention is divided into (workgroups per head). */
export const SPLITS = 8;
/** Floats per (head, split) attention partial: 64 numerators, the max and the sum. */
export const PART_STRIDE = 66;

const head = (sg) => `${sg ? 'enable subgroups;\n' : ''}${STRUCT_P}`;

const BLOB = `
fn ww(i: u32) -> u32 { return wq[i >> 2u][i & 3u]; }
fn wf(i: u32) -> f32 { return bitcast<f32>(ww(i)); }
fn wh(i: u32) -> f32 { return unpack2x16float(ww(i >> 1u))[i & 1u]; }
fn i8x4(u: u32) -> vec4<f32> {
  let i = bitcast<i32>(u);
  return vec4<f32>(f32(extractBits(i, 0u, 8u)), f32(extractBits(i, 8u, 8u)), f32(extractBits(i, 16u, 8u)), f32(extractBits(i, 24u, 8u)));
}
fn h8(q: vec4<u32>) -> array<vec4<f32>, 2> {
  return array<vec4<f32>, 2>(vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y)), vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w)));
}`;

const ACT = `
// erf from the Numerical Recipes Chebyshev fit of erfc (fractional error < 1.2e-7)
fn erf_(z: f32) -> f32 {
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.5 * a);
  let r = t * exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return select(r - 1.0, 1.0 - r, z >= 0.0);
}
fn gelu(v: f32) -> f32 { return 0.5 * v * (1.0 + erf_(v * 0.70710678118654752)); }
fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }`;

/** `wsum` / `wmax` over a whole workgroup of `wg` threads (a power of 2, at least 32), the
 *  result in every thread. Called from uniform control flow only. */
function wgReduce(sg, wg) {
  if (sg) {
    const n = wg / 32;
    const red = (name, op, init) => `
fn ${name}(v: f32, li: u32) -> f32 {
  let s = ${op === 'max' ? 'subgroupMax' : 'subgroupAdd'}(v);
  if ((li & 31u) == 0u) { rsh[li >> 5u] = s; }
  workgroupBarrier();
  var t = ${init};
  for (var i = 0u; i < ${n}u; i++) { t = ${op === 'max' ? 'max(t, rsh[i])' : 't + rsh[i]'}; }
  workgroupBarrier();
  return t;
}`;
    return `var<workgroup> rsh: array<f32, ${n}>;${red('wsum', 'add', '0.0')}${red('wmax', 'max', '-3.0e38')}`;
  }
  const red = (name, op) => `
fn ${name}(v: f32, li: u32) -> f32 {
  rsh[li] = v;
  workgroupBarrier();
  for (var s = ${wg / 2}u; s > 0u; s >>= 1u) {
    if (li < s) { rsh[li] = ${op === 'max' ? 'max(rsh[li], rsh[li + s])' : 'rsh[li] + rsh[li + s]'}; }
    workgroupBarrier();
  }
  let t = rsh[0];
  workgroupBarrier();
  return t;
}`;
  return `var<workgroup> rsh: array<f32, ${wg}>;${red('wsum', 'add')}${red('wmax', 'max')}`;
}

/** `lsum`: the sum over each 32-thread group of a 128-thread workgroup, in each of its threads. */
function laneReduce(sg) {
  if (sg) return 'fn lsum(v: f32, li: u32) -> f32 { return subgroupAdd(v); }';
  return `
var<workgroup> lsh: array<f32, 128>;
fn lsum(v: f32, li: u32) -> f32 {
  lsh[li] = v;
  workgroupBarrier();
  for (var s = 16u; s > 0u; s >>= 1u) {
    if ((li & 31u) < s) { lsh[li] = lsh[li] + lsh[li + s]; }
    workgroupBarrier();
  }
  let t = lsh[li & ~31u];
  workgroupBarrier();
  return t;
}`;
}

const act = (a, v) => (a === 'gelu' ? `gelu(${v})` : a === 'silu' ? `silu(${v})` : v);
const STOP = `if (st[${ST.stop}] != 0u) { return; }`;
/** Codec work past the end of speech the LM found. */
const PAST_END = `if (st[${ST.end}] != 0u && st[${ST.cpos}] >= st[${ST.end}]) { return; }`;

// ---------------------------------------------------------------------------------------------
// The flow LM's decode step

/**
 * One GEMV for the one-row step, y[n] = epilogue(sum_k x[k] w[n][k]). A workgroup is four
 * groups of 32 threads, each computing one output row from 16-byte weight words.
 *
 * pro: `none` (x as is), `ln` (LayerNorm of x over 512, optionally affine, optionally the
 * adaLN modulation `x * (1 + scale) + shift`, optionally keeping the normalized row in `aux`),
 * or `combine` (the attention splits' partials merged into the attention output).
 * Epilogue: `bias`, `actv`, and `res`: `none`, `add` (y += v), `gate` (y += gate[n] * v).
 */
export function gemv({ sg, K, N, wmode, pro = 'none', affine = false, modulate = false, store = false, eps = '1e-5', bias = false, actv = 'none', res = 'none' }) {
  const W = wmode === 'q8' ? 16 : 8; // weights per 16-byte word
  const U = K / W; // words per row
  if (U % 32 !== 0) throw new Error(`gemv K=${K} ${wmode}: a row must be a multiple of 32 words`);
  const K4 = K / 4;
  if (pro !== 'none' && K !== 512) throw new Error('the prologues are written for 512 inputs');
  let prologue;
  if (pro === 'ln') {
    prologue = `
  let v = x[p.xo / 4u + li];
  let mean = wsum(v.x + v.y + v.z + v.w, li) / 512.0;
  let d = v - vec4<f32>(mean);
  let inv = inverseSqrt(wsum(dot(d, d), li) / 512.0 + ${eps});
  var o = d * inv;
  ${affine ? 'o = o * bitcast<vec4<f32>>(wq[p.lg / 4u + li]) + bitcast<vec4<f32>>(wq[p.lb / 4u + li]);' : ''}
  ${modulate ? 'o = o * (vec4<f32>(1.0) + md[p.sc / 4u + li]) + md[p.sh / 4u + li];' : ''}
  ${store ? 'if (wid.x == 0u) { let b = li * 4u; aux[b] = o.x; aux[b + 1u] = o.y; aux[b + 2u] = o.z; aux[b + 3u] = o.w; }' : ''}
  xs[li] = o;
  workgroupBarrier();`;
  } else if (pro === 'combine') {
    prologue = `
  let h = li / 16u; let d = (li % 16u) * 4u;
  var mx = -3.0e38;
  for (var s = 0u; s < ${SPLITS}u; s++) { mx = max(mx, part[(h * ${SPLITS}u + s) * ${PART_STRIDE}u + 64u]); }
  var num = vec4<f32>(0.0); var den = 0.0;
  for (var s = 0u; s < ${SPLITS}u; s++) {
    let b = (h * ${SPLITS}u + s) * ${PART_STRIDE}u;
    let f = exp(part[b + 64u] - mx);
    num += vec4<f32>(part[b + d], part[b + d + 1u], part[b + d + 2u], part[b + d + 3u]) * f;
    den += part[b + 65u] * f;
  }
  xs[li] = num / den;
  workgroupBarrier();`;
  } else {
    prologue = `
  for (var i = li; i < ${K4}u; i += 128u) { xs[i] = x[p.xo / 4u + i]; }
  workgroupBarrier();`;
  }
  const dot8 = wmode === 'q8'
    ? `let q = wq[p.wo + n * ${U}u + u];
      let sc = wh(p.so * 2u + n * ${K / 32}u + u / 2u);
      acc += sc * (dot(i8x4(q.x), xs[u * 4u]) + dot(i8x4(q.y), xs[u * 4u + 1u]) + dot(i8x4(q.z), xs[u * 4u + 2u]) + dot(i8x4(q.w), xs[u * 4u + 3u]));`
    : `let h = h8(wq[p.wo + n * ${U}u + u]);
      acc += dot(h[0], xs[u * 2u]) + dot(h[1], xs[u * 2u + 1u]);`;
  const resid = { none: '', add: 'v += y[p.yo + n];', gate: 'v = y[p.yo + n] + md[(p.ga + n) / 4u][(p.ga + n) % 4u] * v;' }[res];
  return `${head(sg)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> md: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> part: array<f32>;
@group(0) @binding(5) var<storage, read_write> aux: array<f32>;
@group(0) @binding(6) var<storage, read> st: array<u32>;
@group(0) @binding(7) var<uniform> p: P;
var<workgroup> xs: array<vec4<f32>, ${K4}>;
${BLOB}
${ACT}
${pro === 'ln' ? wgReduce(sg, 128) : ''}
${laneReduce(sg)}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  ${STOP}
  let lane = li & 31u;
  ${prologue}
  let n = min(wid.x * 4u + (li >> 5u), ${N - 1}u);
  var acc = 0.0;
  for (var i = 0u; i < ${U / 32}u; i++) {
    let u = lane + i * 32u;
    ${dot8}
  }
  let t = lsum(acc, li);
  if (lane == 0u && wid.x * 4u + (li >> 5u) < ${N}u) {
    var v = t;
    ${bias ? 'v += wf(p.bo + n);' : ''}
    v = ${act(actv, 'v')};
    ${resid}
    y[p.yo + n] = v;
  }
}`;
}

/** Workgroups a `gemv` of N rows needs. */
export const gemvGroups = (N) => Math.ceil(N / 4);

/**
 * Decode attention for the new row: workgroup (split, head), 64 threads. It rotates q
 * (interleaved pairs, as `rope_i`), and when its key range holds the new position it rotates k
 * and writes k and v to the f16 cache; then it scans its keys with an online softmax and
 * writes an unnormalized partial for the out-projection's prologue to merge. The cache is one
 * row of 256 words per position, `[pos][head][64]` as f16 pairs; `p.ko` is the layer's offset.
 */
export function attention({ sg, theta = '10000.0' }) {
  return `${head(sg)}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> kc: array<u32>;
@group(0) @binding(2) var<storage, read_write> vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> part: array<f32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<f32, 64>;
var<workgroup> sc: array<f32, 64>;
${wgReduce(sg, 64)}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  ${STOP}
  let s = wid.x; let h = wid.y;
  let pos = st[${ST.pos}];
  let npos = pos + 1u;
  let chunk = (npos + ${SPLITS - 1}u) / ${SPLITS}u;
  let lo = s * chunk;
  let hi = min(npos, lo + chunk);
  let hb = h * 64u;
  if (li < 32u) {
    let ang = f32(pos) * pow(${theta}, -f32(li) / 32.0);
    let c = cos(ang); let sn = sin(ang);
    let q0 = qkv[hb + 2u * li]; let q1 = qkv[hb + 2u * li + 1u];
    qs[2u * li] = (q0 * c - q1 * sn) * 0.125;
    qs[2u * li + 1u] = (q0 * sn + q1 * c) * 0.125;
    if (hi == npos && lo < hi) {
      let k0 = qkv[512u + hb + 2u * li]; let k1 = qkv[512u + hb + 2u * li + 1u];
      let w = p.ko + (pos * 512u + hb) / 2u + li;
      kc[w] = pack2x16float(vec2<f32>(k0 * c - k1 * sn, k0 * sn + k1 * c));
      vc[w] = pack2x16float(vec2<f32>(qkv[1024u + hb + 2u * li], qkv[1024u + hb + 2u * li + 1u]));
    }
  }
  storageBarrier();
  workgroupBarrier();
  var m = -3.0e38; var l = 0.0; var acc = 0.0;
  for (var base = lo; base < hi; base += 64u) {
    let j = base + li;
    var sv = -3.0e38;
    if (j < hi) {
      var dsum = 0.0;
      let kb = p.ko + (j * 512u + hb) / 2u;
      for (var w = 0u; w < 32u; w++) {
        let kk = unpack2x16float(kc[kb + w]);
        dsum += kk.x * qs[2u * w] + kk.y * qs[2u * w + 1u];
      }
      sv = dsum;
    }
    let nm = max(m, wmax(sv, li));
    let corr = exp(m - nm);
    let pj = select(0.0, exp(sv - nm), j < hi);
    sc[li] = pj;
    l = l * corr + wsum(pj, li);
    acc *= corr;
    for (var jj = 0u; jj < 64u; jj++) {
      if (base + jj < hi) {
        let vw = unpack2x16float(vc[p.ko + ((base + jj) * 512u + hb) / 2u + li / 2u]);
        acc += sc[jj] * select(vw.x, vw.y, (li & 1u) == 1u);
      }
    }
    m = nm;
    workgroupBarrier();
  }
  let pb = (h * ${SPLITS}u + s) * ${PART_STRIDE}u;
  part[pb + li] = acc;
  if (li == 0u) { part[pb + 64u] = m; part[pb + 65u] = l; }
}`;
}

/** The noise for latent element i of frame f: a counter hash (PCG) into Box-Muller. */
const NOISE_FN = `
fn pcg(v: u32) -> u32 { let s = v * 747796405u + 2891336453u; let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return (w >> 22u) ^ w; }
fn gauss(seed: u32, f: u32, i: u32) -> f32 {
  let a = pcg(seed ^ pcg(f * 32u + i));
  let b = pcg(a ^ 0x9e3779b9u);
  let u1 = (f32(a >> 8u) + 0.5) / 16777216.0;
  let u2 = f32(b >> 8u) / 16777216.0;
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}`;

/** The flow net's input projection (32 -> 512, f16) from noise it draws, scaled by
 *  sqrt(temperature), and keeps in `cur` as the sample the solver step adds to. */
export function inproj() {
  return `${STRUCT_P}
@group(0) @binding(0) var<storage, read_write> cur: array<f32>;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> z: array<f32, 32>;
${BLOB}
${NOISE_FN}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  ${STOP}
  if (li < 32u) {
    z[li] = gauss(st[${ST.seed}], st[${ST.frame}], li) * bitcast<f32>(st[${ST.scale}]);
    if (wid.x == 0u) { cur[li] = z[li]; }
  }
  workgroupBarrier();
  let n = wid.x * 256u + li;
  var acc = wf(p.bo + n);
  for (var u = 0u; u < 4u; u++) {
    let h = h8(wq[p.wo + n * 4u + u]);
    acc += dot(h[0], vec4<f32>(z[u * 8u], z[u * 8u + 1u], z[u * 8u + 2u], z[u * 8u + 3u])) +
           dot(h[1], vec4<f32>(z[u * 8u + 4u], z[u * 8u + 5u], z[u * 8u + 6u], z[u * 8u + 7u]));
  }
  y[n] = acc;
}`;
}

/** The step's end, one workgroup of 256: the end-of-speech logit from the final-normed row,
 *  the latent kept for the codec, the end-of-speech rule and the state advanced, and the next
 *  step's input row `x = input_linear(latent) + conditions`. */
export function finish({ sg }) {
  return `${head(sg)}
@group(0) @binding(0) var<storage, read> cur: array<f32>;
@group(0) @binding(1) var<storage, read> tout: array<f32>;
@group(0) @binding(2) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> x: array<f32>;
@group(0) @binding(4) var<storage, read_write> lats: array<f32>;
@group(0) @binding(5) var<storage, read_write> st: array<u32>;
@group(0) @binding(6) var<uniform> p: P;
var<workgroup> lat: array<f32, 32>;
var<workgroup> stopw: u32;
${BLOB}
${wgReduce(sg, 256)}
fn row(n: u32) -> f32 {
  var acc = wf(p.sh + n);
  for (var u = 0u; u < 4u; u++) {
    let h = h8(wq[p.wo + n * 4u + u]);
    acc += dot(h[0], vec4<f32>(lat[u * 8u], lat[u * 8u + 1u], lat[u * 8u + 2u], lat[u * 8u + 3u])) +
           dot(h[1], vec4<f32>(lat[u * 8u + 4u], lat[u * 8u + 5u], lat[u * 8u + 6u], lat[u * 8u + 7u]));
  }
  return acc;
}
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) li: u32) {
  // st is read-write here, so its value is not known to be uniform: broadcast it explicitly.
  if (li == 0u) { stopw = st[${ST.stop}]; }
  if (workgroupUniformLoad(&stopw) != 0u) { return; }
  let f = st[${ST.frame}];
  if (li < 32u) { lat[li] = cur[li]; lats[f * 32u + li] = cur[li]; }
  let logit = wsum(tout[li] * wf(p.lg + li) + tout[li + 256u] * wf(p.lg + li + 256u), li) + wf(p.lb);
  x[li] = row(li);
  x[li + 256u] = row(li + 256u);
  if (li == 0u) {
    let nf = f + 1u;
    var end = st[${ST.end}];
    if (end == 0u && logit > bitcast<f32>(p.a0)) { end = nf + st[${ST.after}]; st[${ST.end}] = end; }
    st[${ST.frame}] = nf;
    st[${ST.pos}] = st[${ST.pos}] + 1u;
    if ((end != 0u && nf >= end) || nf >= st[${ST.limit}]) { st[${ST.stop}] = 1u; }
  }
}`;
}

// ---------------------------------------------------------------------------------------------
// Many rows: the prompt prefill and the codec

/** LayerNorm over 512 channels, one workgroup per row, affine, eps 1e-5. Rows at `p.ab`. */
export function layerNorm({ sg, gate = false }) {
  return `${head(sg)}
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
@group(0) @binding(4) var<uniform> p: P;
${wgReduce(sg, 128)}
@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  ${gate ? PAST_END : ''}
  let v = x[(p.ab + wid.x * 512u) / 4u + li];
  let mean = wsum(v.x + v.y + v.z + v.w, li) / 512.0;
  let d = v - vec4<f32>(mean);
  let inv = inverseSqrt(wsum(dot(d, d), li) / 512.0 + 1e-5);
  y[wid.x * 128u + li] = d * inv * bitcast<vec4<f32>>(wq[p.lg / 4u + li]) + bitcast<vec4<f32>>(wq[p.lb / 4u + li]);
}`;
}

/**
 * C[m][n] = epilogue(sum_k A[m][k] W[n][k]): 64x64 tiles, K in steps of 32, f32 A, q8 or f16 W.
 * A row m starts at float `ab + m * arow` and is K floats long. Output row `m * phases +
 * phase` (wid.z is the phase) is at float `yb + row * N`. Rows: `p.M`, or `st.rows` for a
 * prompt (`rowsFromState`). res: none | add (y = v + r[...]) | addy (y += v) | ls (y += s[n] v).
 * eluA applies ELU to A as it is loaded. gate: codec work, skipped past the end of speech.
 */
export function gemm({ K, N, wmode = 'f16', phases = 1, eluA = false, bias = false, act: a = 'none', res = 'none', rowsFromState = false, gate = false }) {
  if (K % 32) throw new Error(`gemm K=${K}`);
  const loadA = eluA ? 'let v0 = select(exp(v) - vec4<f32>(1.0), v, v > vec4<f32>(0.0));' : 'let v0 = v;';
  // Each tile thread writes whole vec4s, four rows (or columns) at one k: on Metal, threads
  // writing different components of one workgroup vec4 overwrite each other's.
  const loadW = wmode === 'q8'
    ? `for (var i = 0u; i < 4u; i++) {
        let n = tn + g * 4u + i;
        if (n < ${N}u) {
          let q = wq[p.wo + n * ${K / 16}u + k0 / 16u + kq / 4u][kq % 4u];
          ld[i] = i8x4(q) * wh(p.so * 2u + n * ${K / 32}u + k0 / 32u);
        }
      }`
    : `for (var i = 0u; i < 4u; i++) {
        let n = tn + g * 4u + i;
        if (n < ${N}u) {
          let q = wq[p.wo + p.wph * ph + n * ${K / 8}u + k0 / 8u + kq / 2u];
          ld[i] = select(vec4<f32>(unpack2x16float(q.x), unpack2x16float(q.y)), vec4<f32>(unpack2x16float(q.z), unpack2x16float(q.w)), (kq & 1u) == 1u);
        }
      }`;
  const resid = { none: '', add: `v += r[p.rb + orow * ${N}u + n];`, addy: 'v += y[o];', ls: 'v = y[o] + wf(p.lso + n) * v;' }[res];
  return `${STRUCT_P}
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> r: array<f32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> As: array<vec4<f32>, 512>; // [32 k][16 groups of 4 rows]
var<workgroup> Ws: array<vec4<f32>, 512>; // [32 k][16 groups of 4 columns]
${BLOB}
${ACT}
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  ${gate ? PAST_END : ''}
  let M = ${rowsFromState ? `st[${ST.rows}]` : 'p.M'};
  let tm = wid.y * 64u; let tn = wid.x * 64u; let ph = wid.z;
  if (tm >= M) { return; }
  let li = lid.y * 16u + lid.x;
  var acc: array<vec4<f32>, 4>;
  for (var k0 = 0u; k0 < ${K}u; k0 += 32u) {
    // Threads 0-127 load A, 128-255 W: group g of 4 rows (columns), k from 4 * kq.
    let g = (li % 128u) / 8u; let kq = li % 8u;
    var ld: array<vec4<f32>, 4>;
    if (li < 128u) {
      for (var i = 0u; i < 4u; i++) {
        let m = tm + g * 4u + i;
        var v = vec4<f32>(0.0);
        if (m < M) { v = a[(p.ab + m * p.arow + k0) / 4u + kq]; }
        ${loadA}
        ld[i] = v0;
      }
    } else {
      ${loadW}
    }
    for (var c = 0u; c < 4u; c++) {
      let t = vec4<f32>(ld[0][c], ld[1][c], ld[2][c], ld[3][c]);
      if (li < 128u) { As[(kq * 4u + c) * 16u + g] = t; } else { Ws[(kq * 4u + c) * 16u + g] = t; }
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 32u; kk++) {
      let av = As[kk * 16u + lid.y];
      let wv = Ws[kk * 16u + lid.x];
      acc[0] += av.x * wv; acc[1] += av.y * wv; acc[2] += av.z * wv; acc[3] += av.w * wv;
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    let m = tm + lid.y * 4u + i;
    if (m >= M) { continue; }
    let orow = m * ${phases}u + ph;
    for (var j = 0u; j < 4u; j++) {
      let n = tn + lid.x * 4u + j;
      if (n >= ${N}u) { continue; }
      var v = acc[i][j];
      ${bias ? 'v += wf(p.bo + n);' : ''}
      v = ${act(a, 'v')};
      let o = p.yb + orow * ${N}u + n;
      ${resid}
      y[o] = v;
    }
  }
}`;
}

/** Prompt rows: RoPE on q and k of `st.rows` rows at positions `st.pos + m`, q (scaled by
 *  1/8) to `qo`, k and v to the LM's f16 cache. One workgroup per row. */
export function promptRope({ theta = '10000.0' } = {}) {
  return `${STRUCT_P}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> qo: array<f32>;
@group(0) @binding(2) var<storage, read_write> kc: array<u32>;
@group(0) @binding(3) var<storage, read_write> vc: array<u32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wid.x; let t = li % 32u; let h = li / 32u;
  if (m >= st[${ST.rows}]) { return; }
  let pos = st[${ST.pos}] + m;
  let ang = f32(pos) * pow(${theta}, -f32(t) / 32.0);
  let c = cos(ang); let s = sin(ang);
  let b = m * 1536u + h * 64u + 2u * t;
  let q0 = qkv[b]; let q1 = qkv[b + 1u];
  let k0 = qkv[b + 512u]; let k1 = qkv[b + 513u];
  let o = m * 512u + h * 64u + 2u * t;
  qo[o] = (q0 * c - q1 * s) * 0.125; qo[o + 1u] = (q0 * s + q1 * c) * 0.125;
  let w = p.ko + (pos * 512u + h * 64u) / 2u + t;
  kc[w] = pack2x16float(vec2<f32>(k0 * c - k1 * s, k0 * s + k1 * c));
  vc[w] = pack2x16float(vec2<f32>(qkv[b + 1024u], qkv[b + 1025u]));
}`;
}

/** Prompt rows: causal attention of row m (position st.pos + m) over the f16 cache from
 *  position 0, one workgroup per (row, head). */
export function promptAttention({ sg }) {
  return `${head(sg)}
@group(0) @binding(0) var<storage, read> qo: array<f32>;
@group(0) @binding(1) var<storage, read> kc: array<u32>;
@group(0) @binding(2) var<storage, read> vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<f32, 64>;
var<workgroup> sc: array<f32, 64>;
${wgReduce(sg, 64)}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let m = wid.x; let h = wid.y;
  if (m >= st[${ST.rows}]) { return; }
  let hi = st[${ST.pos}] + m + 1u;
  let hb = h * 64u;
  qs[li] = qo[m * 512u + hb + li];
  workgroupBarrier();
  var mx = -3.0e38; var l = 0.0; var acc = 0.0;
  for (var base = 0u; base < hi; base += 64u) {
    let j = base + li;
    var sv = -3.0e38;
    if (j < hi) {
      let kb = p.ko + (j * 512u + hb) / 2u;
      var d = 0.0;
      for (var w = 0u; w < 32u; w++) { let kk = unpack2x16float(kc[kb + w]); d += kk.x * qs[2u * w] + kk.y * qs[2u * w + 1u]; }
      sv = d;
    }
    let nm = max(mx, wmax(sv, li));
    let corr = exp(mx - nm);
    let pj = select(0.0, exp(sv - nm), j < hi);
    sc[li] = pj;
    l = l * corr + wsum(pj, li);
    acc *= corr;
    for (var jj = 0u; jj < 64u; jj++) {
      if (base + jj < hi) {
        let vw = unpack2x16float(vc[p.ko + ((base + jj) * 512u + hb) / 2u + li / 2u]);
        acc += sc[jj] * select(vw.x, vw.y, (li & 1u) == 1u);
      }
    }
    mx = nm;
    workgroupBarrier();
  }
  o[m * 512u + hb + li] = acc / l;
}`;
}

/** Prompt rows: past the prompt, the next KV position. */
export function promptAdvance() {
  return `${STRUCT_P}
@group(0) @binding(0) var<storage, read_write> st: array<u32>;
@group(0) @binding(1) var<uniform> p: P;
@compute @workgroup_size(1)
fn main() { st[${ST.pos}] = st[${ST.pos}] + st[${ST.rows}]; }`;
}

// ---------------------------------------------------------------------------------------------
// The Mimi decoder (codec.js)

const RING = 512;
const WINDOW = 250;
export const CODEC_RING = RING;

/** Codec input: 16 rows per frame of `p.M / 16` frames from `st.cpos`, into rows at `yb`. The
 *  latent denormalization, the quantizer's 32 -> 512 projection (f16, no bias) and the
 *  depthwise transposed-conv upsample (kernel 32, stride 16, f32), whose overlap needs only
 *  the previous frame's latent. */
export function codecIn() {
  return `${STRUCT_P}
@group(0) @binding(0) var<storage, read> lats: array<f32>;
@group(0) @binding(1) var<storage, read> wq: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> st: array<u32>;
@group(0) @binding(4) var<uniform> p: P;
${BLOB}
fn proj(f: u32, c: u32) -> f32 {
  var acc = 0.0;
  for (var u = 0u; u < 4u; u++) {
    let h = h8(wq[p.wo + c * 4u + u]);
    for (var e = 0u; e < 8u; e++) {
      let i = u * 8u + e;
      acc += h[e / 4u][e % 4u] * (lats[f * 32u + i] * wf(p.lg + i) + wf(p.lb + i));
    }
  }
  return acc;
}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  ${PAST_END}
  let c = g.x % 512u; let row = g.x / 512u;
  if (row >= p.M) { return; }
  let f = st[${ST.cpos}] + row / 16u; let j = row % 16u;
  var v = proj(f, c) * wf(p.bo + c * 32u + j);
  if (f > 0u) { v += proj(f - 1u, c) * wf(p.bo + c * 32u + j + 16u); }
  y[p.yb + row * 512u + c] = v;
}`;
}

/** Codec transformer: RoPE on q and k of M rows at 200 Hz position 16 * cpos + m; q to `qo`,
 *  k and v to an f32 ring of RING positions. */
export function ringRope() {
  return `${STRUCT_P}
@group(0) @binding(0) var<storage, read> qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> qo: array<f32>;
@group(0) @binding(2) var<storage, read_write> kr: array<f32>;
@group(0) @binding(3) var<storage, read_write> vr: array<f32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  ${PAST_END}
  let t = g.x % 32u; let h = (g.x / 32u) % 8u; let m = g.x / 256u;
  if (m >= p.M) { return; }
  let pos = st[${ST.cpos}] * 16u + m;
  let ang = f32(pos) * pow(10000.0, -f32(t) / 32.0);
  let c = cos(ang); let s = sin(ang);
  let b = m * 1536u + h * 64u + 2u * t;
  let q0 = qkv[b]; let q1 = qkv[b + 1u];
  let k0 = qkv[b + 512u]; let k1 = qkv[b + 513u];
  let o = m * 512u + h * 64u + 2u * t;
  qo[o] = (q0 * c - q1 * s) * 0.125; qo[o + 1u] = (q0 * s + q1 * c) * 0.125;
  let w = p.ko + (pos % ${RING}u) * 512u + h * 64u + 2u * t;
  kr[w] = k0 * c - k1 * s; kr[w + 1u] = k0 * s + k1 * c;
  vr[w] = qkv[b + 1024u]; vr[w + 1u] = qkv[b + 1025u];
}`;
}

/** Codec transformer: windowed causal attention, one workgroup per (row, head), keys from
 *  pos - (WINDOW - 1) to pos. */
export function ringAttention({ sg }) {
  return `${head(sg)}
@group(0) @binding(0) var<storage, read> qo: array<f32>;
@group(0) @binding(1) var<storage, read> kr: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> vr: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<storage, read> st: array<u32>;
@group(0) @binding(5) var<uniform> p: P;
var<workgroup> qs: array<vec4<f32>, 16>;
var<workgroup> sc: array<f32, 64>;
${wgReduce(sg, 64)}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  ${PAST_END}
  let m = wid.x; let h = wid.y;
  let pos = st[${ST.cpos}] * 16u + m;
  let lo = select(0u, pos + 1u - ${WINDOW}u, pos + 1u > ${WINDOW}u);
  let hi = pos + 1u;
  if (li < 16u) { let b = m * 512u + h * 64u + li * 4u; qs[li] = vec4<f32>(qo[b], qo[b + 1u], qo[b + 2u], qo[b + 3u]); }
  workgroupBarrier();
  var mx = -3.0e38; var l = 0.0; var acc = 0.0;
  for (var base = lo; base < hi; base += 64u) {
    let j = base + li;
    var sv = -3.0e38;
    if (j < hi) {
      let kb = (p.ko + (j % ${RING}u) * 512u + h * 64u) / 4u;
      var d = 0.0;
      for (var e = 0u; e < 16u; e++) { d += dot(kr[kb + e], qs[e]); }
      sv = d;
    }
    let nm = max(mx, wmax(sv, li));
    let corr = exp(mx - nm);
    let pj = select(0.0, exp(sv - nm), j < hi);
    sc[li] = pj;
    l = l * corr + wsum(pj, li);
    acc *= corr;
    for (var jj = 0u; jj < 64u; jj++) {
      if (base + jj < hi) { acc += sc[jj] * vr[p.ko + ((base + jj) % ${RING}u) * 512u + h * 64u + li]; }
    }
    mx = nm;
    workgroupBarrier();
  }
  o[m * 512u + h * 64u + li] = acc / l;
}`;
}

/** Moves each stage's last ctx rows to its front for the next chunk. `stages`: [{ctx, ch, rate}]
 *  in binding order, chunk of T rows at 200 Hz. Not gated: it binds all eight storage buffers
 *  a stage may have, and moving rows past the end of speech is harmless. */
export function shift(stages, T) {
  const binds = stages.map((s, i) => `@group(0) @binding(${i}) var<storage, read_write> b${i}: array<f32>;`).join('\n');
  const n = Math.max(...stages.map((s) => s.ctx * s.ch));
  const body = stages.map((s, i) => `if (i < ${s.ctx * s.ch}u) { b${i}[i] = b${i}[${s.rate * T * s.ch}u + i]; }`).join('\n  ');
  return `${STRUCT_P}
${binds}
@group(0) @binding(${stages.length}) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= ${n}u) { return; }
  ${body}
}`;
}
