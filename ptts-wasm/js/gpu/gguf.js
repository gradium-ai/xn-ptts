// A small GGUF reader: the header, and each tensor's dims (fastest first, as GGUF stores them),
// type and bytes. Enough for the checkpoints `quantize` writes; no metadata is interpreted
// beyond the alignment.

/** ggml type ids this engine reads. */
export const GGML = { F32: 0, F16: 1, Q8_0: 8, BF16: 30 };
const TYPE_SIZE = { [GGML.F32]: 4, [GGML.F16]: 2, [GGML.BF16]: 2 };

/**
 * @param {Uint8Array} buf the whole file
 * @returns {{ tensors: Map<string, { dims: number[], type: number, off: number, n: number }>, dataStart: number, buf: Uint8Array }}
 */
export function parseGguf(buf) {
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let p = 0;
  const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
  const str = () => { const n = u64(); const s = new TextDecoder().decode(buf.subarray(p, p + n)); p += n; return s; };
  if (buf.length < 8 || u32() !== 0x46554747) throw new Error('the weights are not a GGUF file');
  const version = u32();
  if (version < 2) throw new Error(`GGUF version ${version} is not supported`);
  const nTensors = u64();
  const nKv = u64();
  const scalar = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  const value = (t) => {
    if (t === 8) return str();
    if (t === 9) { const it = u32(); const n = u64(); for (let i = 0; i < n; i++) value(it); return undefined; }
    if (!(t in scalar)) throw new Error(`GGUF metadata type ${t} is not supported`);
    const v = t === 4 ? dv.getUint32(p, true) : undefined;
    p += scalar[t];
    return v;
  };
  const kv = {};
  for (let i = 0; i < nKv; i++) { const k = str(); kv[k] = value(u32()); }
  const tensors = new Map();
  for (let i = 0; i < nTensors; i++) {
    const name = str();
    const nd = u32();
    const dims = [];
    for (let d = 0; d < nd; d++) dims.push(u64());
    const type = u32();
    const off = u64();
    tensors.set(name, { dims, type, off, n: dims.reduce((a, b) => a * b, 1) });
  }
  const align = Number(kv['general.alignment'] ?? 32);
  return { tensors, dataStart: Math.ceil(p / align) * align, buf };
}

/** The bytes of tensor `name`. */
export function tensorBytes(g, name) {
  const t = g.tensors.get(name);
  if (!t) throw new Error(`the checkpoint has no tensor ${name}`);
  const bytes = t.type === GGML.Q8_0 ? (t.n / 32) * 34 : TYPE_SIZE[t.type] * t.n;
  if (!bytes) throw new Error(`tensor ${name} has unsupported type ${t.type}`);
  const at = g.dataStart + t.off;
  return g.buf.subarray(at, at + bytes);
}

/**
 * The tensors whose names start with one of `prefixes`, as a safetensors file (row-major shapes,
 * the reverse of GGUF's dims), for the Rust side, which needs a few small ones (the conditioners,
 * the speaker projection) and must not be handed the whole checkpoint. f32, f16 and bf16 only.
 */
export function safetensorsOf(g, prefixes, exclude = []) {
  const picked = [...g.tensors.keys()].filter((n) => prefixes.some((p) => n.startsWith(p)) && !exclude.some((p) => n.startsWith(p)));
  const dtype = { [GGML.F32]: 'F32', [GGML.F16]: 'F16', [GGML.BF16]: 'BF16' };
  const header = {};
  const parts = [];
  let at = 0;
  for (const name of picked) {
    const t = g.tensors.get(name);
    if (!dtype[t.type]) throw new Error(`tensor ${name} has type ${t.type}, expected f32, f16 or bf16`);
    const bytes = tensorBytes(g, name);
    header[name] = { dtype: dtype[t.type], shape: [...t.dims].reverse(), data_offsets: [at, at + bytes.length] };
    parts.push(bytes);
    at += bytes.length;
  }
  let json = new TextEncoder().encode(JSON.stringify(header));
  const pad = (8 - (json.length % 8)) % 8;
  if (pad) json = new TextEncoder().encode(JSON.stringify(header) + ' '.repeat(pad));
  const out = new Uint8Array(8 + json.length + at);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  let o = 8 + json.length;
  for (const b of parts) { out.set(b, o); o += b.length; }
  return out;
}
