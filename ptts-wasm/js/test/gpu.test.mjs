// The WebGPU engine's host-side logic, without a browser or a GPU: which checkpoints it takes,
// which reductions it picks, the GGUF reading, the weight conversions and the WGSL generators.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chooseReductions, seed32, unsupportedConfig, unsupportedWeights } from '../gpu/support.js';
import { GGML, parseGguf, safetensorsOf, tensorBytes } from '../gpu/gguf.js';
import { fromHalf, tensorF32, toHalf } from '../gpu/weights.js';
import * as K from '../gpu/kernels.js';
import { settleDevice } from '../device.js';

/** The parts of `gradium/phonon-31m-en`'s config.json the engine reads. */
const PHONON_31M = {
  model_id: { sig: '5f0a2aab', epoch: 200, mimi_sig: 'b77c7ffa', mimi_epoch: 150 },
  lsd_decode_steps: 1,
  eos_threshold: 4.0,
  flow_lm: {
    d_model: 512, num_heads: 8, num_layers: 6, dim_feedforward: 2048, max_period: 10000.0, n_bins: 4000,
    lut_dim: 512, flow_dim: 512, flow_depth: 6, ldim: 32, time_rms_norm: 'rms', gelu: 'erf',
  },
  mimi: {
    channels: 1, sample_rate: 24000, frame_rate: 12.5, dimension: 512, quantizer_dimension: 32,
    quantizer_output_dimension: 512, n_filters: 64, n_residual_layers: 1, ratios: [6, 5, 4], kernel_size: 7,
    last_kernel_size: 3, residual_kernel_size: 3, dilation_base: 2, compress: 2, transformer_d_model: 512,
    transformer_num_heads: 8, transformer_num_layers: 2, transformer_layer_scale: 0.01, transformer_context: 250,
    transformer_max_period: 10000.0, transformer_dim_feedforward: 2048, gelu: 'erf',
  },
};
const with_ = (patch) => {
  const c = structuredClone(PHONON_31M);
  patch(c);
  return c;
};

test('the published Phonon checkpoints run on WebGPU', () => {
  assert.equal(unsupportedConfig(PHONON_31M), null);
  // The 40M models differ only in depth.
  assert.equal(unsupportedConfig(with_((c) => { c.flow_lm.num_layers = 9; })), null);
});

test('configs the kernels do not cover are refused, naming what is missing', () => {
  const cases = [
    [(c) => { c.lsd_decode_steps = 2; }, /lsd_decode_steps 2/],
    [(c) => { c.flow_lm.flow_type = 'drifting'; }, /drifting flow head/],
    [(c) => { c.flow_lm.flow_type = 'pointwise'; }, /pointwise flow head/],
    [(c) => { c.flow_lm.num_kv_heads = 2; }, /grouped-query attention/],
    [(c) => { c.flow_lm.gelu = 'tanh'; }, /flow LM's tanh GELU/],
    // A pocket-tts export: no model_id, so tanh by default, as TTSConfig::default_gelu says.
    [(c) => { delete c.model_id; delete c.flow_lm.gelu; }, /flow LM's tanh GELU/],
    [(c) => { c.mimi.gelu = 'tanh'; }, /Mimi decoder's tanh GELU/],
    [(c) => { c.flow_lm.z_dim = 32; }, /z_proj/],
    [(c) => { c.flow_lm.attn_dim = 256; }, /attn_dim 256/],
    [(c) => { c.flow_lm.d_model = 768; }, /width of 768|width 768/],
    [(c) => { c.flow_lm.lut_dim = 128; }, /low-rank text table/],
    [(c) => { c.prefix_sinks = [['speaker_wavs', {}]]; }, /prefix sinks/],
    [(c) => { c.audio_delay = 2; }, /audio delay/],
    [(c) => { c.voices = [{ name: 'a' }]; }, /baked-in voices/],
    [(c) => { c.mimi.decoder_dimension = 256; }, /Mimi decoder of width 256/],
    [(c) => { c.mimi.ratios = [8, 6, 5, 4]; }, /Mimi ratios/],
    [(c) => { c.mimi.transformer_num_layers = 8; }, /transformer_num_layers 8/],
  ];
  for (const [patch, re] of cases) {
    const why = unsupportedConfig(with_(patch));
    assert.ok(why, `expected a refusal for ${patch}`);
    assert.match(why, re);
    assert.match(why, /^the WebGPU engine does not run /);
  }
  assert.match(unsupportedConfig({}), /no flow_lm or mimi/);
});

test('the transformer linears must be q8_0', () => {
  const tensors = new Map();
  for (let l = 0; l < 6; l++) {
    for (const w of ['self_attn.in_proj', 'self_attn.out_proj', 'linear1', 'linear2']) {
      tensors.set(`flow_lm.transformer.layers.${l}.${w}.weight`, { type: GGML.Q8_0 });
    }
  }
  assert.equal(unsupportedWeights({ tensors }, PHONON_31M), null);
  tensors.set('flow_lm.transformer.layers.3.linear1.weight', { type: 12 }); // q4_K
  assert.match(unsupportedWeights({ tensors }, PHONON_31M), /reads q8_0 weights, and flow_lm.transformer.layers.3.linear1.weight has ggml type 12/);
  tensors.delete('flow_lm.transformer.layers.0.linear2.weight');
  assert.match(unsupportedWeights({ tensors }, PHONON_31M), /no tensor flow_lm.transformer.layers.0.linear2.weight/);
});

test('subgroup kernels only on 32-wide subgroups', () => {
  const adapter = (features, info) => ({ features: new Set(features), info });
  assert.deepEqual(chooseReductions(adapter(['subgroups'], { subgroupMinSize: 32, subgroupMaxSize: 32 })), { subgroups: true, reason: '32-wide subgroups' });
  assert.equal(chooseReductions(adapter([], { subgroupMinSize: 32, subgroupMaxSize: 32 })).subgroups, false);
  // SwiftShader reports 4; AMD can run 32 or 64; Intel 8 to 32; older Chrome says nothing.
  for (const info of [{ subgroupMinSize: 4, subgroupMaxSize: 4 }, { subgroupMinSize: 32, subgroupMaxSize: 64 }, { subgroupMinSize: 8, subgroupMaxSize: 32 }, {}, undefined]) {
    const r = chooseReductions(adapter(['subgroups'], info));
    assert.equal(r.subgroups, false);
    assert.match(r.reason, /wide, not 32/);
  }
});

test('seed32 folds a u64 into the 32 bits the noise hash takes', () => {
  assert.equal(seed32(0n), 0);
  assert.equal(seed32(1n), 1);
  assert.equal(seed32(1n << 32n), 1);
  assert.equal(seed32(4242424242424242n), seed32(4242424242424242n));
  assert.notEqual(seed32(4242424242424242n), seed32(4242424242424243n));
  assert.ok(seed32((1n << 64n) - 1n) >= 0);
});

test('settleDevice: a refusal is an error under webgpu, the CPU with a warning under auto', () => {
  const choice = { device: 'webgpu', reason: 'requested' };
  assert.deepEqual(settleDevice({ requested: 'webgpu', choice, unsupported: null }), choice);
  assert.deepEqual(settleDevice({ requested: 'webgpu', choice, unsupported: 'the WebGPU engine does not run x' }), { error: 'the WebGPU engine does not run x' });
  const auto = { device: 'webgpu', reason: "'auto', and the browser has a WebGPU adapter" };
  assert.deepEqual(settleDevice({ requested: 'auto', choice: auto, unsupported: 'the WebGPU engine does not run x' }), {
    device: 'cpu', reason: 'the WebGPU engine does not run x', warning: 'the WebGPU engine does not run x; generating on the CPU instead',
  });
  const cpu = { device: 'cpu', reason: 'default' };
  assert.equal(settleDevice({ requested: undefined, choice: cpu, unsupported: 'x' }), cpu);
});

/** A GGUF v3 file holding `tensors` ([name, dims (fastest first), type, bytes]). */
function gguf(tensors) {
  const enc = new TextEncoder();
  const parts = [];
  const u32 = (v) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); parts.push(b); };
  const u64 = (v) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); parts.push(b); };
  const str = (s) => { const b = enc.encode(s); u64(b.length); parts.push(b); };
  u32(0x46554747); u32(3); u64(tensors.length); u64(1);
  str('general.alignment'); u32(4); u32(32);
  let off = 0;
  const offsets = [];
  for (const [name, dims, type, bytes] of tensors) {
    str(name); u32(dims.length); for (const d of dims) u64(d); u32(type); u64(off);
    offsets.push(off);
    off += Math.ceil(bytes.length / 32) * 32;
  }
  const head = parts.reduce((n, p) => n + p.length, 0);
  const start = Math.ceil(head / 32) * 32;
  const out = new Uint8Array(start + off);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  tensors.forEach(([, , , bytes], i) => out.set(bytes, start + offsets[i]));
  return out;
}

const bf16 = (values) => {
  const f = new Float32Array(values);
  const u = new Uint32Array(f.buffer);
  return new Uint8Array(Uint16Array.from(u, (x) => x >>> 16).buffer);
};

test('parseGguf reads dims, types and data; tensorF32 widens bf16', () => {
  const w = [1, -2, 0.5, 3.25, 7, -8];
  const g = parseGguf(gguf([
    ['flow_lm.condition_provider.conditioners.num_speakers.embed.weight', [3, 2], GGML.BF16, bf16(w)],
    ['flow_lm.emb_mean', [2], GGML.F32, new Uint8Array(new Float32Array([0.25, -1]).buffer)],
  ]));
  const t = g.tensors.get('flow_lm.condition_provider.conditioners.num_speakers.embed.weight');
  assert.deepEqual(t.dims, [3, 2]);
  assert.equal(t.type, GGML.BF16);
  assert.deepEqual([...tensorF32(g, 'flow_lm.condition_provider.conditioners.num_speakers.embed.weight')], w);
  assert.deepEqual([...tensorF32(g, 'flow_lm.emb_mean')], [0.25, -1]);
  assert.equal(tensorBytes(g, 'flow_lm.emb_mean').length, 8);
  assert.throws(() => parseGguf(new TextEncoder().encode('{"__metadata__":1}')), /not a GGUF file/);
});

test('safetensorsOf hands Rust only the tensors asked for, with row-major shapes', () => {
  const g = parseGguf(gguf([
    ['flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight', [2, 3], GGML.BF16, bf16([1, 2, 3, 4, 5, 6])],
    ['flow_lm.condition_provider.conditioners.transcript_in_segment.embed.weight', [2, 2], GGML.BF16, bf16([1, 2, 3, 4])],
    ['flow_lm.transformer.layers.0.linear1.weight', [32, 1], GGML.F32, new Uint8Array(128)],
  ]));
  const prefix = 'flow_lm.condition_provider.conditioners.';
  const st = safetensorsOf(g, [prefix], [`${prefix}transcript_in_segment.`]);
  const n = Number(new DataView(st.buffer).getBigUint64(0, true));
  assert.equal(n % 8, 0);
  const header = JSON.parse(new TextDecoder().decode(st.subarray(8, 8 + n)));
  assert.deepEqual(Object.keys(header), ['flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight']);
  const t = header['flow_lm.condition_provider.conditioners.speaker_wavs.output_proj.weight'];
  assert.deepEqual(t, { dtype: 'BF16', shape: [3, 2], data_offsets: [0, 12] });
  assert.equal(st.length, 8 + n + 12);
});

test('f16 conversion rounds to nearest and keeps bf16 values exact', () => {
  for (const v of [0, 1, -1, 0.5, 65504, -2.5, 1e-5, 6.1e-5, 3.14159]) {
    const back = fromHalf(toHalf(v));
    assert.ok(Math.abs(back - v) <= Math.abs(v) * 2 ** -11 + 6e-8, `${v} -> ${back}`);
  }
  assert.equal(fromHalf(toHalf(1e6)), Infinity);
  // bf16 values in f16's range keep every mantissa bit: 8 of them against f16's 11.
  const u = new Uint32Array(1);
  const f = new Float32Array(u.buffer);
  for (const bits of [0x3f80, 0xbf81, 0x3c23, 0x4120, 0x38d1]) {
    u[0] = bits << 16;
    assert.equal(fromHalf(toHalf(f[0])), f[0]);
  }
});

test('every kernel generates for both reduction styles', () => {
  for (const sg of [true, false]) {
    const shaders = [
      K.gemv({ sg, K: 512, N: 1536, wmode: 'q8', pro: 'ln', affine: true }),
      K.gemv({ sg, K: 512, N: 512, wmode: 'q8', pro: 'combine', res: 'add' }),
      K.gemv({ sg, K: 2048, N: 512, wmode: 'q8', res: 'add' }),
      K.gemv({ sg, K: 512, N: 512, wmode: 'f16', pro: 'ln', modulate: true, bias: true, res: 'gate' }),
      K.attention({ sg }),
      K.finish({ sg }),
      K.layerNorm({ sg }),
      K.promptAttention({ sg }),
      K.ringAttention({ sg }),
    ];
    for (const s of shaders) {
      assert.equal(s.includes('enable subgroups;'), sg);
      assert.equal(/subgroup(Add|Max)\(/.test(s), sg);
    }
  }
  for (const s of [K.gemm({ K: 512, N: 1536, wmode: 'q8', rowsFromState: true }), K.gemm({ K: 1024, N: 256, phases: 6, eluA: true, bias: true, gate: true }), K.inproj(), K.codecIn(), K.ringRope(), K.promptRope(), K.promptAdvance()]) {
    assert.ok(!s.includes('subgroup'));
    // No language extension is required: the int8 unpack uses extractBits.
    assert.ok(!s.includes('requires'));
  }
  assert.throws(() => K.gemv({ sg: true, K: 48, N: 8, wmode: 'q8' }), /multiple of 32 words/);
});
