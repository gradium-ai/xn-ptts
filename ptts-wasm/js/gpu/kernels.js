// The WGSL kernels of the WebGPU engine, as generators: each export returns the source of one
// compute shader for one configuration, with its sizes, flags and epsilons baked in as
// constants. dispatch.js compiles them with entry point `main` and an explicit layout of bind
// group 0: the storage bindings in the order lm.js and codec.js list them, then the uniform
// block `P` (one 256-byte slot per dispatch, fields in `P_FIELDS` order).
//
// Units. The weight blob (weights.js) is read as 16-byte words: `wo` counts 16-byte words, and
// every other blob offset (`so`, `bo`, `lg`, `lb`, `lso`, `finish`'s `sh`) counts u32 words.
// int8 weights are unpacked with shifts and f16 with unpack2x16float, so no optional feature
// but subgroups is ever needed. Activations are f32 throughout; the LM's KV cache holds f16
// pairs, one u32 word per pair of channels.
//
// Reductions. With `sg` (subgroups exactly 32 wide, support.js) a 32-invocation group sums with
// subgroupAdd / subgroupMax, taking `local_invocation_index >> 5` as its subgroup. Without it,
// the same groups exchange through workgroup memory, with a barrier between every write and
// the reads of it, so any subgroup width works.
//
// Workgroup memory on Apple GPUs. A write of one component of a workgroup vector may be done as
// a read-modify-write of the whole vector there, so every workgroup array element here is
// written whole, by one invocation, between two barriers: vec4 tiles are filled one whole vec4
// per invocation, and arrays that invocations fill a word each are declared with scalar
// elements. Storage writes follow the same rule (an f16 pair is packed by one invocation).
//
//   gemv        the LM step's matrix-vector products: each 32-invocation group owns one to four
//               output rows (gemvShape), each invocation streaming 16-byte words of a row
//               (adjacent invocations, adjacent words) against input values it holds in
//               registers. The LayerNorm prologue is computed in registers by every group, so
//               it needs no barrier; the attention merge (`combine`) goes through workgroup
//               memory once.
//   attention   split-KV decode attention: workgroup (split, head) scores up to 256 keys at
//               once, one per invocation, then sums values with each invocation owning 8
//               channels of 8 keys; the partials are merged in the out-projection's prologue.
//               The prompt's attention works the same way; Mimi's, over 256-byte f32 keys,
//               gives each key to 8 invocations instead (attentionBySlice).
//   gemm        a 64 x 64 output tile per workgroup. Tiles of more than 32 rows stage A and W
//               in 32-wide K slabs (64 x 32 floats each, the whole 16 KiB), each invocation
//               computing 4 rows x 4 columns; tiles of up to 16 or 32 rows stage only A, in
//               slabs of 256 or 128, and read W straight into registers, the invocations
//               dividing K for each group of 2 or 4 columns and summing through workgroup
//               memory; a GEMM of at most 4 columns gives each row to 4 invocations.

/** Word indices of the state buffer `st` (engine.js describes each slot). */
export const ST = { pos: 0, frame: 1, end: 2, stop: 3, seed: 4, scale: 5, after: 6, limit: 7, cpos: 8, rows: 9 };
/** Length of the state buffer in u32 words. */
export const ST_WORDS = 16;
/** The uniform block: one u32 per field, in this order (`a0` holds the bits of an f32). */
export const P_FIELDS = ['wo', 'so', 'bo', 'lg', 'lb', 'sh', 'sc', 'ga', 'xo', 'yo', 'ko', 'a0', 'M', 'ab', 'arow', 'wph', 'yb', 'rb', 'lso', 'p0'];
/** Bytes of the uniform binding. */
export const P_BYTES = P_FIELDS.length * 4;
/** Key ranges one head's decode attention is cut into: it runs on [SPLITS, 8] workgroups. */
export const SPLITS = 8;
/** Floats per (head, split) partial: 64 unnormalized output sums, the running max, the sum of
 *  the weights, and two of padding so that every partial starts on 16 bytes. */
export const PART_STRIDE = 68;
/** Positions in each Mimi layer's key and value rings. */
export const CODEC_RING = 512;

// ---------------------------------------------------------------------------------------------
// WGSL shared by the kernels

const PARAMS = `struct Params {\n  ${P_FIELDS.map((f) => `${f}: u32,`).join(' ')}\n}`;
const STATE = Object.entries(ST).map(([k, v]) => `const ST_${k.toUpperCase()} = ${v}u;`).join('\n');

/**
 * The top of a shader: the enable, the uniform struct, the state slots and the bindings.
 * `binds` lists the storage bindings in layout order as [kind, name, element type]; a null name
 * leaves that binding undeclared. The uniform block follows them.
 */
function header(sg, binds) {
  const out = sg ? ['enable subgroups;'] : [];
  out.push(PARAMS, STATE, '// "No key yet": finite, so that subtracting it never meets an infinity.', 'const SENT = -3.0e38;');
  binds.forEach(([kind, name, type], i) => {
    if (name) out.push(`@group(0) @binding(${i}) var<storage, ${kind === 'ro' ? 'read' : 'read_write'}> ${name}: array<${type}>;`);
  });
  out.push(`@group(0) @binding(${binds.length}) var<uniform> p: Params;`);
  return out.join('\n');
}

// The blob, bound as `W: array<vec4<u32>>`: a word, an f32, four f32 from a 16-byte boundary,
// and the f16 with half index i.
const BLOB = `
fn bw(i: u32) -> u32 { return W[i >> 2u][i & 3u]; }
fn bf(i: u32) -> f32 { return bitcast<f32>(W[i >> 2u][i & 3u]); }
fn bf4(i: u32) -> vec4<f32> { return bitcast<vec4<f32>>(W[i >> 2u]); }
fn bh(i: u32) -> f32 { return unpack2x16float(W[i >> 3u][(i >> 1u) & 3u])[i & 1u]; }`;

// A 16-byte word as weights: its first or last four f16, or four int8 of one u32.
const UNPACK = `
fn hlo(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn hhi(w: vec4<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.z), unpack2x16float(w.w)); }
fn i8q(w: u32) -> vec4<f32> {
  let i = bitcast<i32>(w);
  return vec4<f32>(vec4<i32>(i << 24u, i << 16u, i << 8u, i) >> vec4<u32>(24u));
}`;

// GELU in its exact form: erf from Numerical Recipes' erfcc (fractional error below 1.2e-7).
const GELU = `
fn erfnr(z: f32) -> f32 {
  let a = abs(z);
  let t = 1.0 / (1.0 + 0.5 * a);
  let r = t * exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
    + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return select(r - 1.0, 1.0 - r, z >= 0.0);
}
fn gelu(v: f32) -> f32 { return 0.5 * v * (1.0 + erfnr(v * 0.70710678118654752)); }`;

const SILU = `
fn silu(v: f32) -> f32 { return v / (1.0 + exp(min(-v, 80.0))); }`;

const ELU = `
fn elu4(v: vec4<f32>) -> vec4<f32> { return select(exp(min(v, vec4<f32>(0.0))) - 1.0, v, v > vec4<f32>(0.0)); }`;

/** An f32 as a WGSL literal that parses back to the same value. */
const lit = (v) => {
  const s = Math.fround(v).toPrecision(9);
  return /[.e]/.test(s) ? s : `${s}.0`;
};

/**
 * RoPE on interleaved pairs: the 32 inverse frequencies theta^(-t/32), rounded to f32 here, and
 * `rot`, which turns a pair by the f32 angle `a`. The angle is brought into [-pi, pi] against
 * 2 pi split into two f32 (Cody and Waite) so that sin and cos stay accurate at positions in the
 * thousands, whatever the GPU's own range reduction does.
 */
function rope(theta) {
  const th = Number(theta);
  if (!(th > 1)) throw new Error(`RoPE base ${theta} is not a number above 1`);
  const om = Array.from({ length: 32 }, (_, t) => lit(th ** (-t / 32)));
  return `
var<private> OMEGA: array<f32, 32> = array<f32, 32>(${om.join(', ')});
fn rot(x: vec2<f32>, a: f32) -> vec2<f32> {
  let k = round(a * 0.15915494309189535);
  let r = fma(-k, -1.7484556e-7, fma(-k, 6.2831854820251465, a));
  let c = cos(r);
  let s = sin(r);
  return vec2<f32>(x.x * c - x.y * s, x.x * s + x.y * c);
}`;
}

/** Sum over the 32 invocations `lid >> 5` names; every one of them gets the result. */
function groupSum(sg, wg) {
  if (sg) return 'fn gsum(v: f32, lid: u32) -> f32 { return subgroupAdd(v); }';
  return `
var<workgroup> gred: array<f32, ${wg}>;
fn gsum(v: f32, lid: u32) -> f32 {
  gred[lid] = v;
  workgroupBarrier();
  let b = lid & ~31u;
  var s = 0.0;
  for (var i = 0u; i < 32u; i++) { s += gred[b + i]; }
  workgroupBarrier();
  return s;
}`;
}

/**
 * Sum (`op` 'Add') or max ('Max') over a workgroup of `wg` invocations, every one getting the
 * result. Its workgroup memory may be written again after the next barrier.
 */
function workgroupReduce(sg, wg, op, name) {
  const f = op === 'Add' ? (a, b) => `${a} + ${b}` : (a, b) => `max(${a}, ${b})`;
  const groups = wg / 32;
  if (sg) {
    return `
var<workgroup> ${name}m: array<f32, ${groups}>;
fn ${name}(v: f32, lid: u32) -> f32 {
  let s = subgroup${op}(v);
  if ((lid & 31u) == 0u) { ${name}m[lid >> 5u] = s; }
  workgroupBarrier();
  var r = ${name}m[0];
  for (var i = 1u; i < ${groups}u; i++) { r = ${f('r', `${name}m[i]`)}; }
  return r;
}`;
  }
  return `
var<workgroup> ${name}m: array<f32, ${wg}>;
fn ${name}(v: f32, lid: u32) -> f32 {
  ${name}m[lid] = v;
  workgroupBarrier();
  for (var s = ${wg / 2}u; s > 0u; s >>= 1u) {
    if (lid < s) { ${name}m[lid] = ${f(`${name}m[lid]`, `${name}m[lid + s]`)}; }
    workgroupBarrier();
  }
  return ${name}m[0];
}`;
}

const MAIN = '@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_index) lid: u32';
const STOPPED = 'if (st[ST_STOP] != 0u) { return; }';
const PAST_END = 'if (st[ST_END] != 0u && st[ST_CPOS] >= st[ST_END]) { return; }';
const XYZW = ['x', 'y', 'z', 'w'];

// ---------------------------------------------------------------------------------------------
// gemv

/**
 * How a GEMV of N rows is cut, a function of N alone (gemvGroups has only N): `rw` rows per
 * 32-invocation group, `groups` groups per workgroup. Many rows go several to a group, which
 * then loads its input once for all of them; few rows go one to a group so that they still
 * spread over the GPU.
 */
function gemvShape(N) {
  if (N >= 4096 && N % 8 === 0) return { rw: 2, groups: 4 };
  if (N >= 1024 && N % 16 === 0) return { rw: 4, groups: 4 };
  if (N >= 256 && N % 4 === 0) return { rw: 1, groups: 4 };
  return { rw: 1, groups: 1 };
}

/** Workgroups of a GEMV with N output rows (a 1-D grid). */
export function gemvGroups(N) {
  const { rw, groups } = gemvShape(N);
  return Math.ceil(N / (rw * groups));
}

/**
 * y = act(W x^ + b), then a residual: one row of the LM step through a q8 or f16 matrix, with
 * the input prologue (as is, LayerNorm with optional affine / adaLN modulation, or the merge of
 * the decode attention's partials) and the epilogue fused.
 *
 * Each 32-invocation group owns `rw` adjacent rows (gemvShape). Lane j takes each row's
 * 16-byte words j, j + 32, ..., so the group reads 512 contiguous bytes per load, and holds the
 * 4 (q8) or 2 (f16) input quads each word multiplies; with a LayerNorm prologue those are
 * exactly the 16 inputs the lane loads, normalizes and modulates, the statistics coming from
 * two group sums. All the weight loads are issued before the prologue.
 */
export function gemv({ sg, K, N, wmode, pro = 'none', affine = false, modulate = false, store = false, eps = '1e-5', bias = false, actv = 'none', res = 'none' }) {
  const q8 = wmode === 'q8';
  if (!q8 && wmode !== 'f16') throw new Error(`gemv: unknown weight format ${wmode}`);
  const kw = q8 ? 16 : 8; // weights per 16-byte word
  if (!(K > 0) || K % (32 * kw) !== 0) throw new Error(`gemv: a row of ${K} ${wmode} weights is not a multiple of 32 words of 16 bytes`);
  if (pro !== 'none' && K !== 512) throw new Error(`gemv: the '${pro}' prologue is for K = 512, not ${K}`);
  if (!['none', 'ln', 'combine'].includes(pro)) throw new Error(`gemv: unknown prologue ${pro}`);
  const { rw: RW, groups } = gemvShape(N);
  const WG = 32 * groups;
  const WPR = K / kw; // words per row
  const WL = WPR / 32; // words per lane
  const QW = kw / 4; // input quads per word
  const md = modulate || res === 'gate';
  const head = header(sg, [
    ['ro', pro === 'combine' ? null : 'xin', 'vec4<f32>'],
    ['ro', 'W', 'vec4<u32>'],
    ['rw', 'y', 'f32'],
    ['ro', md ? 'md' : null, 'vec4<f32>'],
    ['ro', pro === 'combine' ? 'part' : null, 'vec4<f32>'],
    ['rw', store ? 'aux' : null, 'vec4<f32>'],
    ['ro', 'st', 'u32'],
  ]);
  const rows = Array.from({ length: RW }, (_, r) => r);
  const b = [STOPPED, 'let lane = lid & 31u;', `let row = (wg.x * ${groups}u + (lid >> 5u)) * ${RW}u;`];
  for (const r of rows) b.push(`let n${r} = min(row + ${r}u, ${N - 1}u);`);
  for (const r of rows) for (let i = 0; i < WL; i++) b.push(`let w${r}_${i} = W[p.wo + n${r} * ${WPR}u + lane + ${32 * i}u];`);
  if (q8) for (const r of rows) for (let i = 0; i < WL; i++) b.push(`let s${r}_${i} = bh(2u * p.so + n${r} * ${K / 32}u + ((lane + ${32 * i}u) >> 1u));`);
  for (const r of rows) {
    if (bias) b.push(`let bv${r} = bf(p.bo + n${r});`);
    if (res !== 'none') b.push(`let yv${r} = y[p.yo + n${r}];`);
    if (res === 'gate') b.push(`let gv${r} = md[(p.ga + n${r}) >> 2u][(p.ga + n${r}) & 3u];`);
  }
  // The input quads, x<i>_<h>: quad (lane + 32 i) * QW + h of x^.
  const X = [];
  for (let i = 0; i < WL; i++) for (let h = 0; h < QW; h++) X.push({ v: `x${i}_${h}`, q: `q${i}_${h}` });
  for (const { q } of X) {
    const [i, h] = q.slice(1).split('_').map(Number);
    b.push(`let ${q} = (lane + ${32 * i}u) * ${QW}u + ${h}u;`);
  }
  if (pro === 'none') {
    for (const { v, q } of X) b.push(`let ${v} = xin[(p.xo >> 2u) + ${q}];`);
  } else if (pro === 'ln') {
    for (const { v, q } of X) b.push(`let r${v} = xin[(p.xo >> 2u) + ${q}];`);
    b.push(`let mean = gsum(${X.map(({ v }) => `dot(r${v}, vec4<f32>(1.0))`).join(' + ')}, lid) / 512.0;`);
    for (const { v } of X) b.push(`let d${v} = r${v} - mean;`);
    b.push(`let inv = inverseSqrt(gsum(${X.map(({ v }) => `dot(d${v}, d${v})`).join(' + ')}, lid) / 512.0 + ${eps});`);
    for (const { v, q } of X) {
      let e = `d${v} * inv`;
      if (affine) e = `${e} * bf4(p.lg + 4u * ${q}) + bf4(p.lb + 4u * ${q})`;
      if (modulate) e = `(${e}) * (1.0 + md[(p.sc >> 2u) + ${q}]) + md[(p.sh >> 2u) + ${q}]`;
      b.push(`let ${v} = ${e};`);
    }
    if (store) b.push(`if (wg.x == 0u && lid < 32u) {\n    ${X.map(({ v, q }) => `aux[${q}] = ${v};`).join('\n    ')}\n  }`);
  } else {
    // The split-KV merge, once per workgroup: quad e is channels 4e .. 4e + 3, head e / 16.
    b.push(`for (var e = lid; e < 128u; e += ${WG}u) {
    let h = e >> 4u;
    var mx = SENT;
    for (var s = 0u; s < ${SPLITS}u; s++) { mx = max(mx, part[(h * ${SPLITS}u + s) * ${PART_STRIDE / 4}u + 16u].x); }
    var num = vec4<f32>(0.0);
    var den = 0.0;
    for (var s = 0u; s < ${SPLITS}u; s++) {
      let at = (h * ${SPLITS}u + s) * ${PART_STRIDE / 4}u;
      let ml = part[at + 16u];
      let f = exp(ml.x - mx);
      num += f * part[at + (e & 15u)];
      den += f * ml.y;
    }
    xs[e] = num / den;
  }
  workgroupBarrier();`);
    for (const { v, q } of X) b.push(`let ${v} = xs[${q}];`);
  }
  for (const r of rows) {
    const terms = [];
    for (let i = 0; i < WL; i++) {
      if (q8) terms.push(`s${r}_${i} * (${XYZW.map((c, h) => `dot(i8q(w${r}_${i}.${c}), x${i}_${h})`).join(' + ')})`);
      else terms.push(`dot(hlo(w${r}_${i}), x${i}_0) + dot(hhi(w${r}_${i}), x${i}_1)`);
    }
    b.push(`let tot${r} = gsum(${terms.join('\n    + ')}, lid);`);
  }
  if (!['none', 'gelu', 'silu'].includes(actv)) throw new Error(`gemv: unknown activation ${actv}`);
  if (!['none', 'add', 'gate'].includes(res)) throw new Error(`gemv: unknown residual ${res}`);
  const epi = (r) => {
    const e = [`var v = tot${r};`];
    if (bias) e.push(`v += bv${r};`);
    if (actv !== 'none') e.push(`v = ${actv}(v);`);
    if (res === 'add') e.push(`v = yv${r} + v;`);
    else if (res === 'gate') e.push(`v = yv${r} + gv${r} * v;`);
    e.push(`y[p.yo + n${r}] = v;`);
    return `if (lane == ${r}u && row + ${r}u < ${N}u) {\n    ${e.join('\n    ')}\n  }`;
  };
  b.push(...rows.map(epi));
  return `${head}
${BLOB}
${UNPACK}
${actv === 'gelu' ? GELU : ''}${actv === 'silu' ? SILU : ''}
${groupSum(sg, WG)}
${pro === 'combine' ? 'var<workgroup> xs: array<vec4<f32>, 128>;' : ''}
@compute @workgroup_size(${WG})
fn main(${MAIN}) {
  ${b.join('\n  ')}
}
`;
}

// ---------------------------------------------------------------------------------------------
// Attention: the LM's decode and prompt attentions and Mimi's ring attention share one body.

/**
 * One query row of one head against the f16 keys [j0, j1) of the LM cache, 256 at a time with
 * the online softmax. Invocation i scores key jb + i (its 128 bytes, against the query in `q`
 * quads); then invocation (g, c) = (lid >> 3, lid & 7) sums channels 8c .. 8c + 7 of keys
 * jb + g + 32 t, t < 8, whose values it loaded with its key. The 32 key groups are summed at
 * the end, into `o` (64 numerators, invocations 0..63), `mx` and `den`.
 *
 * `row(j)` is the 16-byte index of position j's first word for this head, and `q(i)` is quad i
 * of the scaled query.
 */
function attentionByKey({ row, q }) {
  const s = [];
  for (let w = 0; w < 8; w++) s.push(`let k${w} = K[${row('jk')} + ${w}u];`);
  s.push(`let sc = ${Array.from({ length: 8 }, (_, w) => `dot(hlo(k${w}), ${q(2 * w)}) + dot(hhi(k${w}), ${q(2 * w + 1)})`).join('\n      + ')};`);
  const v = [];
  const acc = [];
  for (let t = 0; t < 8; t++) {
    v.push(`let jv${t} = min(jb + g + ${32 * t}u, j1 - 1u);`, `let v${t} = V[${row(`jv${t}`)} + c];`);
    acc.push(`let p${t} = ps[g + ${32 * t}u];`, `a0 = fma(vec4<f32>(p${t}), hlo(v${t}), a0);`, `a1 = fma(vec4<f32>(p${t}), hhi(v${t}), a1);`, `ls += p${t};`);
  }
  return `
  let g = lid >> 3u;
  let c = lid & 7u;
  var mx = SENT;
  var lt = 0.0;
  var o0 = vec4<f32>(0.0);
  var o1 = vec4<f32>(0.0);
  for (var jb = j0; jb < j1; jb += 256u) {
    ${v.join('\n    ')}
    let jk = min(jb + lid, j1 - 1u);
    ${s.join('\n    ')}
    let live = jb + lid < j1;
    let mn = max(mx, bmax(select(SENT, sc, live), lid));
    let al = exp(mx - mn);
    ps[lid] = select(0.0, exp(sc - mn), live);
    workgroupBarrier();
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var ls = 0.0;
    ${acc.join('\n    ')}
    o0 = o0 * al + a0;
    o1 = o1 * al + a1;
    lt = lt * al + ls;
    mx = mn;
  }
  red[2u * lid] = o0;
  red[2u * lid + 1u] = o1;
  if (c == 0u) { lsum[g] = lt; }
  workgroupBarrier();
  var o = 0.0;
  var den = 0.0;
  if (lid < 64u) {
    let at = 2u * (lid >> 3u) + ((lid >> 2u) & 1u);
    for (var gg = 0u; gg < 32u; gg++) {
      o += red[16u * gg + at][lid & 3u];
      den += lsum[gg];
    }
  }`;
}

/**
 * One query row of one head against the f32 keys [j0, j1) of a Mimi ring, 256 at a time with
 * the online softmax. A key is 256 bytes here, so rather than one per invocation, invocation
 * (g, c) = (lid >> 3, lid & 7) owns channels 8c .. 8c + 7 of keys jb + g + 32 t, t < 8: it
 * loads that slice of each key and value (the 8 invocations of a group read a position's
 * channels contiguously), and the slices' dot products are summed across the group (subgroup
 * shuffles, or workgroup memory) into the scores, which every invocation of the group then
 * holds; weights and value sums stay in registers. The 32 key groups are summed at the end, as
 * in attentionByKey.
 *
 * `row(j)` is the 16-byte index of position j's first channel for this head (keys and values
 * alike); `ql` / `qh` are the scaled query's channels 8c .. 8c + 3 and 8c + 4 .. 8c + 7.
 */
function attentionBySlice({ sg, row, ql, qh }) {
  const T8 = [0, 1, 2, 3, 4, 5, 6, 7];
  const L = [];
  for (const t of T8) {
    L.push(`let jk${t} = jb + g + ${32 * t}u;`, `let r${t} = ${row(`min(jk${t}, j1 - 1u)`)};`);
    L.push(`let k${t} = K[r${t} + 2u * c];`, `let l${t} = K[r${t} + 2u * c + 1u];`, `let v${t} = V[r${t} + 2u * c];`, `let u${t} = V[r${t} + 2u * c + 1u];`,
      `let d${t} = dot(k${t}, ql) + dot(l${t}, qh);`);
  }
  if (sg) {
    for (const t of T8) L.push(`var s${t} = d${t} + subgroupShuffleXor(d${t}, 1u);`, `s${t} += subgroupShuffleXor(s${t}, 2u);`, `s${t} += subgroupShuffleXor(s${t}, 4u);`);
  } else {
    for (const t of T8) L.push(`wsc[8u * (g + ${32 * t}u) + c] = d${t};`);
    L.push('workgroupBarrier();');
    for (const t of T8) L.push(`var s${t} = ${T8.map((e) => `wsc[8u * (g + ${32 * t}u) + ${e}u]`).join(' + ')};`);
  }
  for (const t of T8) L.push(`s${t} = select(SENT, s${t}, jk${t} < j1);`);
  L.push('let mn = max(mx, bmax(max(max(max(s0, s1), max(s2, s3)), max(max(s4, s5), max(s6, s7))), lid));', 'let al = exp(mx - mn);');
  L.push('var a0 = vec4<f32>(0.0);', 'var a1 = vec4<f32>(0.0);', 'var ls = 0.0;');
  for (const t of T8) {
    L.push(`let p${t} = exp(s${t} - mn);`, `a0 = fma(vec4<f32>(p${t}), v${t}, a0);`, `a1 = fma(vec4<f32>(p${t}), u${t}, a1);`);
    L.push(`ls += p${t};`);
  }
  return `
  let g = lid >> 3u;
  let c = lid & 7u;
  let ql = ${ql};
  let qh = ${qh};
  var mx = SENT;
  var lt = 0.0;
  var o0 = vec4<f32>(0.0);
  var o1 = vec4<f32>(0.0);
  for (var jb = j0; jb < j1; jb += 256u) {
    ${L.join('\n    ')}
    o0 = o0 * al + a0;
    o1 = o1 * al + a1;
    lt = lt * al + ls;
    mx = mn;
  }
  ${[0, 1, 2, 3].map((e) => `wsc[8u * lid + ${e}u] = o0.${XYZW[e]};\n  wsc[8u * lid + ${e + 4}u] = o1.${XYZW[e]};`).join('\n  ')}
  if (c == 0u) { lsum[g] = lt; }
  workgroupBarrier();
  var o = 0.0;
  var den = 0.0;
  if (lid < 64u) {
    for (var gg = 0u; gg < 32u; gg++) {
      o += wsc[64u * gg + lid];
      den += lsum[gg];
    }
  }`;
}

const ATTN_BY_KEY = `
var<workgroup> ps: array<f32, 256>;
var<workgroup> red: array<vec4<f32>, 512>;
var<workgroup> lsum: array<f32, 32>;`;

const ATTN_BY_SLICE = `
var<workgroup> wsc: array<f32, 2048>;
var<workgroup> lsum: array<f32, 32>;`;

/**
 * Decode attention of the step's new row, for split `wg.x` of head `wg.y`. The split holding
 * position P first writes the new key (rotated) and value into the f16 cache, so that every
 * score reads keys as the cache rounds them, then the partial (unnormalized sums, max, sum of
 * weights) goes to PART for the out-projection's `combine` prologue.
 */
export function attention({ sg, theta = '10000.0' }) {
  const head = header(sg, [
    ['ro', 'qkv', 'vec4<f32>'],
    ['rw', 'K', 'vec4<u32>'],
    ['rw', 'V', 'vec4<u32>'],
    ['rw', 'part', 'f32'],
    ['ro', 'st', 'u32'],
  ]);
  return `${head}
${UNPACK}
${rope(theta)}
${workgroupReduce(sg, 256, 'Max', 'bmax')}
${ATTN_BY_KEY}
var<workgroup> qs: array<vec4<f32>, 16>;
fn rot4(x: vec4<f32>, pos: f32, t: u32) -> vec4<f32> {
  return vec4<f32>(rot(x.xy, pos * OMEGA[t]), rot(x.zw, pos * OMEGA[t + 1u]));
}
fn pk(a: vec4<f32>, b: vec4<f32>) -> vec4<u32> {
  return vec4<u32>(pack2x16float(a.xy), pack2x16float(a.zw), pack2x16float(b.xy), pack2x16float(b.zw));
}
@compute @workgroup_size(256)
fn main(${MAIN}) {
  ${STOPPED}
  let h = wg.y;
  let P = st[ST_POS];
  let cs = (P + ${SPLITS}u) / ${SPLITS}u;
  let j0 = wg.x * cs;
  let j1 = min(P + 1u, j0 + cs);
  // Invocation t < 8 rotates channels 8t .. 8t + 7: the query (scaled by 1 / 8) into qs, and,
  // in the split holding P, the key into the cache, with the value.
  if (lid < 8u) {
    let pos = f32(P);
    let qa = qkv[16u * h + 2u * lid];
    let qb = qkv[16u * h + 2u * lid + 1u];
    qs[2u * lid] = rot4(qa, pos, 4u * lid) * 0.125;
    qs[2u * lid + 1u] = rot4(qb, pos, 4u * lid + 2u) * 0.125;
    if (j0 <= P && P < j1) {
      let at = (p.ko >> 2u) + 64u * P + 8u * h + lid;
      K[at] = pk(rot4(qkv[128u + 16u * h + 2u * lid], pos, 4u * lid), rot4(qkv[128u + 16u * h + 2u * lid + 1u], pos, 4u * lid + 2u));
      V[at] = pk(qkv[256u + 16u * h + 2u * lid], qkv[256u + 16u * h + 2u * lid + 1u]);
    }
  }
  workgroupBarrier();
  storageBarrier();
  ${attentionByKey({ row: (j) => `(p.ko >> 2u) + 64u * ${j} + 8u * h`, q: (i) => `qs[${i}]` })}
  let at = (h * ${SPLITS}u + wg.x) * ${PART_STRIDE}u;
  if (lid < 64u) { part[at + lid] = o; }
  if (lid == 0u) {
    part[at + 64u] = mx;
    part[at + 65u] = den;
  }
}
`;
}

/** Causal attention of prompt row `wg.x`, head `wg.y`, over the f16 cache: keys 0 .. pos + m. */
export function promptAttention({ sg }) {
  const head = header(sg, [
    ['ro', 'pq', 'vec4<f32>'],
    ['ro', 'K', 'vec4<u32>'],
    ['ro', 'V', 'vec4<u32>'],
    ['rw', 'att', 'f32'],
    ['ro', 'st', 'u32'],
  ]);
  return `${head}
${UNPACK}
${workgroupReduce(sg, 256, 'Max', 'bmax')}
${ATTN_BY_KEY}
@compute @workgroup_size(256)
fn main(${MAIN}) {
  let m = wg.x;
  if (m >= st[ST_ROWS]) { return; }
  let h = wg.y;
  let j0 = 0u;
  let j1 = st[ST_POS] + m + 1u;
  ${Array.from({ length: 16 }, (_, i) => `let q${i} = pq[128u * m + 16u * h + ${i}u];`).join('\n  ')}
  ${attentionByKey({ row: (j) => `(p.ko >> 2u) + 64u * ${j} + 8u * h`, q: (i) => `q${i}` })}
  if (lid < 64u) { att[512u * m + 64u * h + lid] = o / den; }
}
`;
}

/** Mimi's windowed causal attention of row `wg.x`, head `wg.y`: the 250 positions up to
 *  16 cpos + m, from the f32 rings. */
export function ringAttention({ sg }) {
  const head = header(sg, [
    ['ro', 'q', 'vec4<f32>'],
    ['ro', 'K', 'vec4<f32>'],
    ['ro', 'V', 'vec4<f32>'],
    ['rw', 'att', 'f32'],
    ['ro', 'st', 'u32'],
  ]);
  const slot = (j) => `(p.ko >> 2u) + 128u * (${j} & ${CODEC_RING - 1}u) + 16u * h`;
  return `${head}
${workgroupReduce(sg, 256, 'Max', 'bmax')}
${ATTN_BY_SLICE}
@compute @workgroup_size(256)
fn main(${MAIN}) {
  ${PAST_END}
  let m = wg.x;
  let h = wg.y;
  let P = 16u * st[ST_CPOS] + m;
  let j0 = max(P, 249u) - 249u;
  let j1 = P + 1u;
  ${attentionBySlice({ sg, row: slot, ql: 'q[128u * m + 16u * h + 2u * (lid & 7u)]', qh: 'q[128u * m + 16u * h + 2u * (lid & 7u) + 1u]' })}
  if (lid < 64u) { att[512u * m + 64u * h + lid] = o / den; }
}
`;
}

// ---------------------------------------------------------------------------------------------
// The flow head's ends: the noise in, the step's state out.

/** PCG hash (one LCG step, then RXS-M-XS) and the Box-Muller Gaussian of element i of frame f. */
const NOISE = `
fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn gauss(seed: u32, f: u32, i: u32) -> f32 {
  let a = pcg(seed ^ pcg(32u * f + i));
  let b = pcg(a ^ 0x9E3779B9u);
  let u1 = (f32(a >> 8u) + 0.5) / 16777216.0;
  let u2 = f32(b >> 8u) / 16777216.0;
  let t = 6.283185307179586 * u2;
  // cos(t) for t in [0, 2 pi), taken as -cos(t - pi) to stay in the range GPUs keep accurate.
  let r = fma(-1.0, -8.742278e-8, t - 3.1415927410125732);
  return sqrt(-2.0 * log(u1)) * -cos(r);
}`;

/** The frame's noise (into CUR) and the flow net's input projection of it (into HF). */
export function inproj() {
  const head = header(false, [
    ['rw', 'cur', 'vec4<f32>'],
    ['ro', 'W', 'vec4<u32>'],
    ['rw', 'hf', 'f32'],
    ['ro', 'st', 'u32'],
  ]);
  return `${head}
${BLOB}
${UNPACK}
${NOISE}
var<workgroup> zs: array<vec4<f32>, 8>;
@compute @workgroup_size(256)
fn main(${MAIN}) {
  ${STOPPED}
  if (lid < 8u) {
    let sd = st[ST_SEED];
    let f = st[ST_FRAME];
    let z = vec4<f32>(gauss(sd, f, 4u * lid), gauss(sd, f, 4u * lid + 1u), gauss(sd, f, 4u * lid + 2u), gauss(sd, f, 4u * lid + 3u))
      * bitcast<f32>(st[ST_SCALE]);
    zs[lid] = z;
    if (wg.x == 0u) { cur[lid] = z; }
  }
  let n = 256u * wg.x + lid;
  let w0 = W[p.wo + 4u * n];
  let w1 = W[p.wo + 4u * n + 1u];
  let w2 = W[p.wo + 4u * n + 2u];
  let w3 = W[p.wo + 4u * n + 3u];
  let b = bf(p.bo + n);
  workgroupBarrier();
  hf[n] = b + dot(hlo(w0), zs[0]) + dot(hhi(w0), zs[1]) + dot(hlo(w1), zs[2]) + dot(hhi(w1), zs[3])
    + dot(hlo(w2), zs[4]) + dot(hhi(w2), zs[5]) + dot(hlo(w3), zs[6]) + dot(hhi(w3), zs[7]);
}
`;
}

/**
 * The step's end, in one workgroup: the latent into the chunk's list, the end-of-speech logit,
 * the next step's input row (input_linear(latent) + conditions), and the state.
 */
export function finish({ sg }) {
  const head = header(sg, [
    ['ro', 'cur', 'vec4<f32>'],
    ['ro', 'tout', 'vec2<f32>'],
    ['ro', 'W', 'vec4<u32>'],
    ['rw', 'x', 'f32'],
    ['rw', 'lats', 'vec4<f32>'],
    ['rw', 'st', 'u32'],
  ]);
  const row = (n) => `bf(p.sh + ${n}) + ${[0, 1, 2, 3].map((j) => `dot(hlo(W[p.wo + 4u * ${n} + ${j}u]), c${2 * j}) + dot(hhi(W[p.wo + 4u * ${n} + ${j}u]), c${2 * j + 1})`).join('\n    + ')}`;
  return `${head}
${BLOB}
${UNPACK}
${workgroupReduce(sg, 256, 'Add', 'wsum')}
var<workgroup> flag: u32;
@compute @workgroup_size(256)
fn main(${MAIN}) {
  // st is read-write here, so the stop flag reaches every invocation through workgroup memory.
  let f = st[ST_FRAME];
  if (lid == 0u) { flag = st[ST_STOP]; }
  if (workgroupUniformLoad(&flag) != 0u) { return; }
  ${Array.from({ length: 8 }, (_, i) => `let c${i} = cur[${i}u];`).join('\n  ')}
  if (lid < 8u) { lats[8u * f + lid] = cur[lid]; }
  let n0 = lid;
  let n1 = lid + 256u;
  x[n0] = ${row('n0')};
  x[n1] = ${row('n1')};
  let t = tout[lid];
  let logit = wsum(t.x * bf(p.lg + 2u * lid) + t.y * bf(p.lg + 2u * lid + 1u), lid) + bf(p.lb);
  if (lid == 0u) {
    let nf = f + 1u;
    var ne = st[ST_END];
    if (ne == 0u && logit > bitcast<f32>(p.a0)) {
      ne = nf + st[ST_AFTER];
      st[ST_END] = ne;
    }
    st[ST_FRAME] = nf;
    st[ST_POS] = st[ST_POS] + 1u;
    if ((ne != 0u && nf >= ne) || nf >= st[ST_LIMIT]) { st[ST_STOP] = 1u; }
  }
}
`;
}

// ---------------------------------------------------------------------------------------------
// Row-wise kernels

/** LayerNorm (eps 1e-5, affine) of row `wg.x`, from float `ab` of the input to row m of the
 *  output; one 32-invocation group per row, four quads each. */
export function layerNorm({ sg, gate = false }) {
  const head = header(sg, [
    ['ro', 'xin', 'vec4<f32>'],
    ['ro', 'W', 'vec4<u32>'],
    ['rw', 'dst', 'vec4<f32>'],
    ['ro', 'st', 'u32'],
  ]);
  const I = [0, 1, 2, 3];
  return `${head}
${BLOB}
${groupSum(sg, 32)}
@compute @workgroup_size(32)
fn main(${MAIN}) {
  ${gate ? PAST_END : ''}
  let m = wg.x;
  let b = (p.ab >> 2u) + 128u * m;
  ${I.map((i) => `let x${i} = xin[b + lid + ${32 * i}u];`).join('\n  ')}
  let mean = gsum(${I.map((i) => `dot(x${i}, vec4<f32>(1.0))`).join(' + ')}, lid) / 512.0;
  ${I.map((i) => `let d${i} = x${i} - mean;`).join('\n  ')}
  let inv = inverseSqrt(gsum(${I.map((i) => `dot(d${i}, d${i})`).join(' + ')}, lid) / 512.0 + 1e-5);
  ${I.map((i) => `dst[128u * m + lid + ${32 * i}u] = d${i} * inv * bf4(p.lg + 4u * (lid + ${32 * i}u)) + bf4(p.lb + 4u * (lid + ${32 * i}u));`).join('\n  ')}
}
`;
}

/** The prompt's RoPE: row `wg.x`'s q (rotated, scaled by 1 / 8) into PQ, and its k (rotated) and
 *  v into the f16 cache at position pos + m. Invocation (h, t) owns pair t of head h. */
export function promptRope({ theta = '10000.0' } = {}) {
  const head = header(false, [
    ['ro', 'qkv', 'vec2<f32>'],
    ['rw', 'pq', 'vec2<f32>'],
    ['rw', 'kc', 'u32'],
    ['rw', 'vc', 'u32'],
    ['ro', 'st', 'u32'],
  ]);
  return `${head}
${rope(theta)}
@compute @workgroup_size(256)
fn main(${MAIN}) {
  let m = wg.x;
  if (m >= st[ST_ROWS]) { return; }
  let P = st[ST_POS] + m;
  let a = f32(P) * OMEGA[lid & 31u];
  let i = 768u * m + lid;
  pq[256u * m + lid] = rot(qkv[i], a) * 0.125;
  let at = p.ko + 256u * P + lid;
  kc[at] = pack2x16float(rot(qkv[i + 256u], a));
  vc[at] = pack2x16float(qkv[i + 512u]);
}
`;
}

/** Mimi's RoPE at the 200 Hz position 16 cpos + m: q into Q, k and v into the rings. */
export function ringRope() {
  const head = header(false, [
    ['ro', 'qkv', 'vec2<f32>'],
    ['rw', 'q', 'vec2<f32>'],
    ['rw', 'kr', 'vec2<f32>'],
    ['rw', 'vr', 'vec2<f32>'],
    ['ro', 'st', 'u32'],
  ]);
  return `${head}
${rope('10000.0')}
@compute @workgroup_size(256)
fn main(${MAIN}) {
  ${PAST_END}
  let m = wg.x;
  if (m >= p.M) { return; }
  let P = 16u * st[ST_CPOS] + m;
  let a = f32(P) * OMEGA[lid & 31u];
  let i = 768u * m + lid;
  q[256u * m + lid] = rot(qkv[i], a) * 0.125;
  let at = (p.ko >> 1u) + 256u * (P & ${CODEC_RING - 1}u) + lid;
  kr[at] = rot(qkv[i + 256u], a);
  vr[at] = qkv[i + 512u];
}
`;
}

/** After a prompt: the position moves past its rows. */
export function promptAdvance() {
  return `${header(false, [['rw', 'st', 'u32']])}
@compute @workgroup_size(1)
fn main() {
  st[ST_POS] = st[ST_POS] + st[ST_ROWS];
}
`;
}

/**
 * Latents to Mimi's 200 Hz rows: denormalize frame f's latent (and f - 1's), project both to
 * 512 channels with the quantizer's output projection, and apply the depthwise transposed
 * convolution (kernel 32, stride 16): row 16 f + j gets tap j of frame f, tap j + 16 of f - 1.
 * A workgroup is half a row, so its two latents are staged once.
 */
export function codecIn() {
  const head = header(false, [
    ['ro', 'lats', 'vec4<f32>'],
    ['ro', 'W', 'vec4<u32>'],
    ['rw', 'tr', 'f32'],
    ['ro', 'st', 'u32'],
  ]);
  return `${head}
${BLOB}
${UNPACK}
var<workgroup> lat: array<vec4<f32>, 16>;
@compute @workgroup_size(256)
fn main(${MAIN}) {
  ${PAST_END}
  let row = wg.x >> 1u;
  if (row >= p.M) { return; }
  let f = st[ST_CPOS] + (row >> 4u);
  let j = row & 15u;
  // lat[0..8): frame f, lat[8..16): frame f - 1 (zeros at the chunk's first frame).
  if (lid < 16u) {
    let i = lid & 7u;
    var v = vec4<f32>(0.0);
    if (lid < 8u) { v = lats[8u * f + i] * bf4(p.lg + 4u * i) + bf4(p.lb + 4u * i); }
    else if (f > 0u) { v = lats[8u * (f - 1u) + i] * bf4(p.lg + 4u * i) + bf4(p.lb + 4u * i); }
    lat[lid] = v;
  }
  let c = 256u * (wg.x & 1u) + lid;
  ${[0, 1, 2, 3].map((w) => `let w${w} = W[p.wo + 4u * c + ${w}u];`).join('\n  ')}
  let u0 = bf(p.bo + 32u * c + j);
  let u1 = bf(p.bo + 32u * c + j + 16u);
  workgroupBarrier();
  var pf = 0.0;
  var pp = 0.0;
  ${[0, 1, 2, 3].map((w) => `pf += dot(hlo(w${w}), lat[${2 * w}]) + dot(hhi(w${w}), lat[${2 * w + 1}]);\n  pp += dot(hlo(w${w}), lat[${8 + 2 * w}]) + dot(hhi(w${w}), lat[${9 + 2 * w}]);`).join('\n  ')}
  tr[p.yb + 512u * row + c] = select(pf * u0, pf * u0 + pp * u1, f > 0u);
}
`;
}

/** Each context-keeping stage's last `ctx` rows (after a chunk of T rows at 200 Hz) to its
 *  front, for the next chunk. 256 element indices per workgroup. */
export function shift(stages, T) {
  if (stages.length > 8) throw new Error(`shift: ${stages.length} stages, more than 8 storage bindings`);
  const head = header(false, stages.map((_, i) => ['rw', `b${i}`, 'f32']));
  const moves = stages.map((s, i) => {
    const n = s.ctx * s.ch;
    return `if (e < ${n}u) { b${i}[e] = b${i}[${s.rate * T * s.ch}u + e]; }`;
  });
  return `${head}
@compute @workgroup_size(256)
fn main(${MAIN}) {
  let e = 256u * wg.x + lid;
  ${moves.join('\n  ')}
}
`;
}

// ---------------------------------------------------------------------------------------------
// gemm

const gemmCache = new Map();

/**
 * Y = epilogue(act(A^ W^T + b)) for a 64 x 64 tile of outputs per workgroup (and phase z), A
 * rows read from float `ab` with stride `arow` (overlapping rows express convolutions), ELU'd
 * as they are read if `eluA`.
 *
 * Tiles of 33 to 64 rows: K in slabs of 32, A (64 x 8 quads) and W (64 columns x 8 quads) staged
 * whole-vec4 by whole-vec4 and dequantized as they are; invocation (ty, tx) computes rows
 * ty + 16 i and columns 4 tx .. 4 tx + 3, as k-quads against the tiles. W rows are swizzled
 * (quad q of column c at q ^ ((c >> 2) & 7)) so that the 8 columns one load phase reads fall
 * in different banks. The next slab's loads are in flight while the current one is computed.
 *
 * Tiles of up to 16 (or 32) rows, the prompt's and the first chunks': only A is staged, 256 (or
 * 128) k at a time, so that W can be read straight from memory, many words in flight per
 * invocation (gemmNarrow). At most 4 columns (the final convolution): gemmThin.
 */
export function gemm({ K, N, wmode = 'f16', phases = 1, eluA = false, bias = false, act = 'none', res = 'none', rowsFromState = false, gate = false }) {
  if (!(K > 0) || K % 32 !== 0) throw new Error(`gemm: K = ${K} is not a multiple of 32`);
  if (wmode !== 'q8' && wmode !== 'f16') throw new Error(`gemm: unknown weight format ${wmode}`);
  if (!['none', 'gelu'].includes(act)) throw new Error(`gemm: unknown activation ${act}`);
  if (!['none', 'add', 'addy', 'ls'].includes(res)) throw new Error(`gemm: unknown residual ${res}`);
  const key = JSON.stringify([K, N, wmode, phases, eluA, bias, act, res, rowsFromState, gate]);
  let code = gemmCache.get(key);
  if (code === undefined) {
    code = gemmSource({ K, N, q8: wmode === 'q8', phases, eluA, bias, act, res, rowsFromState, gate });
    gemmCache.set(key, code);
  }
  return code;
}

function gemmSource(o) {
  const { K, N, q8, phases, eluA, bias, act, res, rowsFromState, gate } = o;
  const vecY = N % 4 === 0;
  const head = header(false, [
    ['ro', 'A', 'vec4<f32>'],
    ['ro', 'W', 'vec4<u32>'],
    ['rw', 'Y', vecY ? 'vec4<f32>' : 'f32'],
    ['ro', res === 'add' ? 'rsd' : null, 'f32'],
    ['ro', 'st', 'u32'],
  ]);
  // Loads, zero past M rows, K or N columns.
  const loads = `
const K = ${K}u;
const N = ${N}u;
fn ldA(m: u32, k: u32, M: u32) -> vec4<f32> {
  if (m >= M || k >= K) { return vec4<f32>(0.0); }
  return ${eluA ? 'elu4(' : ''}A[(p.ab >> 2u) + m * (p.arow >> 2u) + (k >> 2u)]${eluA ? ')' : ''};
}
${q8 ? `fn ldW(n: u32, k: u32, ph: u32) -> vec4<u32> {
  if (n >= N || k >= K) { return vec4<u32>(0u); }
  return W[p.wo + n * ${K / 16}u + (k >> 4u)];
}
fn ldS(n: u32, k: u32) -> f32 {
  if (n >= N || k >= K) { return 0.0; }
  return bh(2u * p.so + n * ${K / 32}u + (k >> 5u));
}` : `fn ldW(n: u32, k: u32, ph: u32) -> vec4<u32> {
  if (n >= N || k >= K) { return vec4<u32>(0u); }
  return W[p.wo + ph * p.wph + n * ${K / 8}u + (k >> 3u)];
}`}
fn swz(q: u32) -> u32 { return q ^ ((q >> 3u) & 3u); }`;
  return `${head}
${BLOB}
${UNPACK}
${act === 'gelu' ? GELU : ''}
${eluA ? ELU : ''}
${loads}
${gemmPut(o, vecY)}
var<workgroup> sm: array<vec4<f32>, 1024>;
${N <= 4 ? gemmThin(o) : `${gemmWide(o)}\n${gemmNarrow(16, 4, o)}\n${gemmNarrow(32, q8 ? 2 : 4, o)}`}
@compute @workgroup_size(256)
fn main(${MAIN}) {
  ${gate ? PAST_END : ''}
  let M = ${rowsFromState ? 'st[ST_ROWS]' : 'p.M'};
  let m0 = 64u * wg.y;
  if (m0 >= M) { return; }
  ${N <= 4 ? 'thin(lid, m0, wg.z, M);' : `let mt = min(64u, M - m0);
  if (mt <= 16u) { nar16(lid, m0, 64u * wg.x, wg.z, mt); }
  else if (mt <= 32u) { nar32(lid, m0, 64u * wg.x, wg.z, mt); }
  else { wide(lid, m0, 64u * wg.x, wg.z, M); }`}
}
`;
}

/** The epilogue for 4 adjacent columns n .. n + 3 of output row `ro` (row m's phase). */
function gemmPut({ N, bias, act, res, phases }, vecY) {
  const PH = phases;
  const one = (v, j) => {
    const n = `n + ${j}u`;
    const s = [`var ${v} = acc.${XYZW[j]};`];
    if (bias) s.push(`${v} += bf(p.bo + ${n});`);
    if (act === 'gelu') s.push(`${v} = gelu(${v});`);
    return s;
  };
  if (vecY) {
    const s = ['if (n >= N) { return; }', 'let o = (p.yb + ro * N + n) >> 2u;'];
    for (let j = 0; j < 4; j++) s.push(...one(`v${j}`, j));
    const v = 'vec4<f32>(v0, v1, v2, v3)';
    if (res === 'none') s.push(`Y[o] = ${v};`);
    else if (res === 'add') s.push(`let r = p.rb + ro * N + n;`, `Y[o] = ${v} + vec4<f32>(rsd[r], rsd[r + 1u], rsd[r + 2u], rsd[r + 3u]);`);
    else if (res === 'addy') s.push(`Y[o] = Y[o] + ${v};`);
    else s.push(`Y[o] = Y[o] + vec4<f32>(bf(p.lso + n), bf(p.lso + n + 1u), bf(p.lso + n + 2u), bf(p.lso + n + 3u)) * ${v};`);
    return `fn put(m: u32, ph: u32, n: u32, acc: vec4<f32>) {\n  let ro = m * ${PH}u + ph;\n  ${s.join('\n  ')}\n}`;
  }
  const s = [];
  for (let j = 0; j < 4; j++) {
    const t = [...one('v', j), `let o = p.yb + ro * N + n + ${j}u;`];
    if (res === 'none') t.push('Y[o] = v;');
    else if (res === 'add') t.push(`Y[o] = v + rsd[p.rb + ro * N + n + ${j}u];`);
    else if (res === 'addy') t.push('Y[o] = Y[o] + v;');
    else t.push(`Y[o] = Y[o] + bf(p.lso + n + ${j}u) * v;`);
    s.push(`if (n + ${j}u < N) {\n    ${t.join('\n    ')}\n  }`);
  }
  return `fn put(m: u32, ph: u32, n: u32, acc: vec4<f32>) {\n  let ro = m * ${PH}u + ph;\n  ${s.join('\n  ')}\n}`;
}

/** At most 4 columns (the final convolution's one): four invocations per row, each summing
 *  every fourth 16-byte word of K, then added through workgroup memory. */
function gemmThin({ N, q8 }) {
  const kw = q8 ? 16 : 8;
  const cols = Array.from({ length: N }, (_, j) => j);
  const term = (j) => (q8
    ? `ldS(${j}u, ${kw}u * w) * (${XYZW.map((c, h) => `dot(i8q(x${j}.${c}), a${h})`).join(' + ')})`
    : `dot(hlo(x${j}), a0) + dot(hhi(x${j}), a1)`);
  return `fn thin(lid: u32, m0: u32, ph: u32, M: u32) {
  let m = m0 + (lid >> 2u);
  var acc = vec4<f32>(0.0);
  for (var w = lid & 3u; w < K / ${kw}u; w += 4u) {
    ${Array.from({ length: kw / 4 }, (_, h) => `let a${h} = ldA(m, ${kw}u * w + ${4 * h}u, M);`).join('\n    ')}
    ${cols.map((j) => `let x${j} = ldW(${j}u, ${kw}u * w, ph);`).join('\n    ')}
    acc += vec4<f32>(${[0, 1, 2, 3].map((j) => (j < N ? term(j) : '0.0')).join(',\n      ')});
  }
  sm[lid] = acc;
  workgroupBarrier();
  if (lid < 64u && m0 + lid < M) {
    put(m0 + lid, ph, 0u, sm[4u * lid] + sm[4u * lid + 1u] + sm[4u * lid + 2u] + sm[4u * lid + 3u]);
  }
}`;
}

/** 64-row tiles: A and W both staged, 32 k per slab. */
function gemmWide({ q8 }) {
  const I = [0, 1, 2, 3];
  const Q = [0, 1, 2, 3, 4, 5, 6, 7];
  // Staging: A rows ar and ar + 32, quad aq; W column wc, words of 8 (f16) or 16 (q8) k.
  const loadW = q8
    ? (k) => [`if (lid < 128u) {`, `  rw = ldW(n0 + wc, ${k} + 16u * wh, ph);`, `  rs = ldS(n0 + wc, ${k});`, '}']
    : (k) => [`rw = ldW(n0 + wc, ${k} + 8u * wh, ph);`];
  const load = (k) => [`ra0 = ldA(m0 + ar, ${k} + 4u * aq, M);`, `ra1 = ldA(m0 + ar + 32u, ${k} + 4u * aq, M);`, ...loadW(k)];
  const stageW = q8
    ? [`if (lid < 128u) {`, ...I.map((h) => `  sm[wt + ((4u * wh + ${h}u) ^ wsw)] = i8q(rw.${XYZW[h]}) * rs;`), '}']
    : ['sm[wt + ((2u * wh) ^ wsw)] = hlo(rw);', 'sm[wt + ((2u * wh + 1u) ^ wsw)] = hhi(rw);'];
  const quad = (q) => [
    `{`,
    `  let pw = ${q}u ^ sx;`,
    ...I.map((j) => `  let w${j} = sm[wb + ${8 * j}u + pw];`),
    ...I.map((k) => `  let t${k} = vec4<f32>(w0.${XYZW[k]}, w1.${XYZW[k]}, w2.${XYZW[k]}, w3.${XYZW[k]});`),
    ...I.flatMap((i) => [
      `  let a${i} = sm[ab + ${128 * i + q}u];`,
      `  c${i} = fma(vec4<f32>(a${i}.x), t0, fma(vec4<f32>(a${i}.y), t1, fma(vec4<f32>(a${i}.z), t2, fma(vec4<f32>(a${i}.w), t3, c${i}))));`,
    ]),
    `}`,
  ];
  return `fn wide(lid: u32, m0: u32, n0: u32, ph: u32, M: u32) {
  let tx = lid & 15u;
  let ty = lid >> 4u;
  let sx = tx & 7u;
  let wb = 512u + 32u * tx;
  let ab = 8u * ty;
  let ar = lid >> 3u;
  let aq = lid & 7u;
  let wc = ${q8 ? '(lid >> 1u) & 63u' : 'lid >> 2u'};
  let wh = lid & ${q8 ? 1 : 3}u;
  let wt = 512u + 8u * wc;
  let wsw = (wc >> 2u) & 7u;
  ${I.map((i) => `var c${i} = vec4<f32>(0.0);`).join('\n  ')}
  var ra0 = vec4<f32>(0.0);
  var ra1 = vec4<f32>(0.0);
  var rw = vec4<u32>(0u);
  ${q8 ? 'var rs = 0.0;' : ''}
  ${load('0u').join('\n  ')}
  for (var k0 = 0u; k0 < K; k0 += 32u) {
    workgroupBarrier();
    sm[8u * ar + aq] = ra0;
    sm[8u * ar + 256u + aq] = ra1;
    ${stageW.join('\n    ')}
    workgroupBarrier();
    if (k0 + 32u < K) {
      ${load('k0 + 32u').join('\n      ')}
    }
    ${Q.flatMap(quad).join('\n    ')}
  }
  ${I.map((i) => `if (ty + ${16 * i}u < M - m0) { put(m0 + ty + ${16 * i}u, ph, n0 + 4u * tx, c${i}); }`).join('\n  ')}
}`;
}

/**
 * Tiles of at most R (16 or 32) rows: A staged R x (4096 / R) at a time, W from memory.
 * Invocation (cg, g) = (lid / G, lid % G) owns columns CT cg .. CT cg + CT - 1 and words g,
 * g + G, ... of each one's slab, G = 4 CT k-groups in all; each A quad it reads from workgroup
 * memory feeds 4 CT multiply-adds.
 */
function gemmNarrow(R, CT, { q8 }) {
  const G = 4 * CT;
  const KS = 4096 / R; // k per slab
  const QS = KS / 4; // quads per A row
  const kw = q8 ? 16 : 8;
  const U = KS / G / kw; // words of each column per invocation per slab
  if (!Number.isInteger(U) || U < 1) throw new Error(`gemm: no ${R}-row tiling with ${CT} columns per invocation`);
  const SL = 64 / R; // reduction slots of 16 R vec4 in the 1024
  const vt = `vec${CT}<f32>`;
  const rows = Array.from({ length: R }, (_, r) => r);
  const us = Array.from({ length: U }, (_, u) => u);
  const js = Array.from({ length: CT }, (_, j) => j);
  const I = [0, 1, 2, 3];
  const load = (k) => [
    ...I.map((i) => `ra${i} = ldA(m0 + ar${i}, ${k} + 4u * aq${i}, M);`),
    ...us.flatMap((u) => js.flatMap((j) => [
      `w${j}_${u} = ldW(nc${j}, ${k} + ${kw}u * (g + ${G * u}u), ph);`,
      ...(q8 ? [`s${j}_${u} = ldS(nc${j}, ${k} + 16u * (g + ${G * u}u));`] : []),
    ])),
  ];
  const quad = (u, h) => {
    const qx = q8 ? `4u * (g + ${G * u}u) + ${h}u` : `2u * (g + ${G * u}u) + ${h}u`;
    const wv = (j) => (q8 ? `i8q(x${j}_${u}.${XYZW[h]}) * y${j}_${u}` : `${h ? 'hhi' : 'hlo'}(x${j}_${u})`);
    return [
      '{',
      `  let pq = swz(${qx});`,
      ...js.map((j) => `  let v${j} = ${wv(j)};`),
      ...I.map((k) => `  let t${k} = ${vt}(${js.map((j) => `v${j}.${XYZW[k]}`).join(', ')});`),
      ...rows.flatMap((r) => [
        `  let a${r} = sm[${r * QS}u + pq];`,
        `  c${r} = fma(${vt}(a${r}.x), t0, fma(${vt}(a${r}.y), t1, fma(${vt}(a${r}.z), t2, fma(${vt}(a${r}.w), t3, c${r}))));`,
      ]),
      '}',
    ];
  };
  const quads = us.flatMap((u) => Array.from({ length: q8 ? 4 : 2 }, (_, h) => quad(u, h)).flat());
  // An invocation's sums as vec4: row r's 4 columns (CT 4), or rows 2 i and 2 i + 1's 2 (CT 2).
  const NV = (R * CT) / 4;
  const sums = Array.from({ length: NV }, (_, i) => (CT === 4 ? [`c${i}`] : [`c${2 * i}`, `c${2 * i + 1}`]));
  const asVec = (i) => (CT === 4 ? sums[i][0] : `vec4<f32>(${sums[i][0]}, ${sums[i][1]})`);
  const addVec = (i, v) => (CT === 4 ? `${sums[i][0]} + ${v}` : `vec4<f32>(${sums[i][0]} + ${v}.xy, ${sums[i][1]} + ${v}.zw)`);
  // The k-groups' sums: the batch of SL groups at the top writes its slots, each lower batch
  // adds its own and writes back, then every invocation sums the SL slots for its outputs.
  const red = [];
  for (let top = G; top > 0; top -= SL) {
    const lo = top - SL;
    red.push('workgroupBarrier();');
    const body = [];
    for (let i = 0; i < NV; i++) {
      const at = `${16 * R}u * (g - ${lo}u) + ${NV}u * cg + ${i}u`;
      if (top === G) body.push(`sm[${at}] = ${asVec(i)};`);
      else body.push(`{ let v = sm[${at}]; sm[${at}] = ${addVec(i, 'v')}; }`);
    }
    red.push(`if (g >= ${lo}u && g < ${top}u) {\n    ${body.join('\n    ')}\n  }`);
  }
  red.push('workgroupBarrier();');
  const outs = [];
  for (let i = 0; i < R / 16; i++) {
    const slot = (s) => (CT === 4
      ? `v += sm[${16 * R * s}u + ${R}u * cq + r];`
      : `{ let x = sm[${16 * R * s}u + ${R}u * cq + (r >> 1u)]; let y = sm[${16 * R * s}u + ${R}u * cq + ${R / 2}u + (r >> 1u)]; v += select(vec4<f32>(x.xy, y.xy), vec4<f32>(x.zw, y.zw), (r & 1u) == 1u); }`);
    outs.push(`{
    let r = (lid >> 4u) + ${16 * i}u;
    var v = vec4<f32>(0.0);
    ${Array.from({ length: SL }, (_, s) => slot(s)).join('\n    ')}
    if (r < mt) { put(m0 + r, ph, n0 + 4u * cq, v); }
  }`);
  }
  return `fn nar${R}(lid: u32, m0: u32, n0: u32, ph: u32, mt: u32) {
  let M = m0 + mt;
  let g = lid & ${G - 1}u;
  let cg = lid / ${G}u;
  let cq = lid & 15u;
  ${js.map((j) => `let nc${j} = n0 + ${CT}u * cg + ${j}u;`).join('\n  ')}
  ${I.map((i) => `let ar${i} = ${(256 * i) / QS}u + lid / ${QS}u;\n  let aq${i} = lid % ${QS}u;\n  let as${i} = ${QS}u * ar${i} + swz(aq${i});`).join('\n  ')}
  ${rows.map((r) => `var c${r} = ${vt}(0.0);`).join('\n  ')}
  ${I.map((i) => `var ra${i} = vec4<f32>(0.0);`).join('\n  ')}
  ${us.flatMap((u) => js.map((j) => `var w${j}_${u} = vec4<u32>(0u);${q8 ? ` var s${j}_${u} = 0.0;` : ''}`)).join('\n  ')}
  ${load('0u').join('\n  ')}
  for (var k0 = 0u; k0 < K; k0 += ${KS}u) {
    workgroupBarrier();
    ${I.map((i) => `sm[as${i}] = ra${i};`).join('\n    ')}
    workgroupBarrier();
    ${us.flatMap((u) => js.map((j) => `let x${j}_${u} = w${j}_${u};${q8 ? ` let y${j}_${u} = s${j}_${u};` : ''}`)).join('\n    ')}
    if (k0 + ${KS}u < K) {
      ${load(`k0 + ${KS}u`).join('\n      ')}
    }
    ${quads.join('\n    ')}
  }
  ${red.join('\n  ')}
  ${outs.join('\n  ')}
}`;
}
