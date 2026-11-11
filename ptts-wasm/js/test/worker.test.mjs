// worker.js itself, under node: which device a load ends up on, what a failed WebGPU start
// leaves behind, and what reaches the page. The wasm module and the WebGPU engine are fakes
// (test/worker/), swapped in by a module hook, with a fake adapter and fetch; device.js,
// support.js and gguf.js are the real ones.

import { test, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { register } from 'node:module';
import { GGML } from '../gpu/gguf.js';

register('./worker/hooks.mjs', import.meta.url);

/** A config the WebGPU engine runs: `gradium/phonon-31m-en`'s, one layer deep. */
const CONFIG = {
  model_id: { sig: '5f0a2aab', epoch: 200 },
  lsd_decode_steps: 1,
  flow_lm: {
    d_model: 512, num_heads: 8, num_layers: 1, dim_feedforward: 2048, max_period: 10000.0, n_bins: 4000,
    lut_dim: 512, flow_dim: 512, flow_depth: 6, ldim: 32, gelu: 'erf',
  },
  mimi: {
    sample_rate: 24000, frame_rate: 12.5, dimension: 512, quantizer_dimension: 32, quantizer_output_dimension: 512,
    n_filters: 64, n_residual_layers: 1, ratios: [6, 5, 4], kernel_size: 7, last_kernel_size: 3,
    residual_kernel_size: 3, dilation_base: 2, compress: 2, transformer_d_model: 512, transformer_num_heads: 8,
    transformer_num_layers: 2, transformer_layer_scale: 0.01, transformer_context: 250,
    transformer_max_period: 10000.0, transformer_dim_feedforward: 2048, gelu: 'erf',
  },
};

/** A GGUF v3 file holding `tensors` ([name, dims (fastest first), type, bytes]). */
function gguf(tensors) {
  const enc = new TextEncoder();
  const parts = [];
  const u32 = (v) => { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, v, true); parts.push(b); };
  const u64 = (v) => { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(v), true); parts.push(b); };
  const str = (s) => { const b = enc.encode(s); u64(b.length); parts.push(b); };
  u32(0x46554747); u32(3); u64(tensors.length); u64(0);
  let off = 0;
  for (const [name, dims, type, bytes] of tensors) {
    str(name); u32(dims.length); for (const d of dims) u64(d); u32(type); u64(off);
    off += Math.ceil(bytes.length / 32) * 32;
  }
  const head = parts.reduce((n, p) => n + p.length, 0);
  const out = new Uint8Array(Math.ceil(head / 32) * 32 + off);
  let at = 0;
  for (const p of parts) { out.set(p, at); at += p.length; }
  return out;
}

const layerWeights = () =>
  gguf(['self_attn.in_proj', 'self_attn.out_proj', 'linear1', 'linear2'].map((w) => [`flow_lm.transformer.layers.0.${w}.weight`, [32, 1], GGML.Q8_0, new Uint8Array(34)]));

const FILES = new Map([
  ['mem:weights', layerWeights()],
  ['mem:tokenizer', new TextEncoder().encode('{}')],
  ['mem:config', new TextEncoder().encode(JSON.stringify(CONFIG))],
  ['mem:config-2steps', new TextEncoder().encode(JSON.stringify({ ...CONFIG, lsd_decode_steps: 2 }))],
  ['mem:voice-a', new TextEncoder().encode('a')],
  ['mem:voice-bad', new TextEncoder().encode('bad')],
]);
const MODEL = {
  weights: { q8: 'mem:weights' },
  tokenizer: 'mem:tokenizer',
  config: 'mem:config',
  voices: { a: 'mem:voice-a', bad: 'mem:voice-bad' },
};

globalThis.fetch = async (url) => {
  globalThis.phononFake.fetches.push(url);
  const bytes = FILES.get(url);
  if (!bytes) return new Response(null, { status: 404 });
  return new Response(bytes.slice(), { headers: { 'content-length': String(bytes.length) } });
};

// The worker's yield, without a real MessagePort, which would keep node from exiting.
globalThis.MessageChannel = class {
  port1 = {};
  port2 = { postMessage: () => setImmediate(() => this.port1.onmessage?.()) };
};

/** A hardware adapter whose device starts, or fails as `phononFake` says. */
const adapter = {
  features: new Set(),
  info: { isFallbackAdapter: false },
  async requestDevice() {
    const fake = globalThis.phononFake;
    fake.log.push('requestDevice');
    if (fake.deviceFails) throw new Error(fake.deviceFails);
    let lose;
    return {
      lost: new Promise((resolve) => (lose = resolve)),
      pushErrorScope() {},
      async popErrorScope() {
        return fake.validation ? { message: fake.validation } : null;
      },
      destroy() {
        fake.log.push('device.destroy');
        lose({ reason: 'destroyed', message: '' });
      },
    };
  },
};

beforeEach(() => {
  globalThis.phononFake = {
    log: [], fetches: [], warnings: [], adapter,
    deviceFails: null, preparerFails: null, engineFails: null, validation: null, cpuFails: null,
  };
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    writable: true,
    value: { hardwareConcurrency: 8, gpu: { requestAdapter: async () => globalThis.phononFake.adapter } },
  });
  console.warn = (...args) => globalThis.phononFake.warnings.push(args.join(' '));
});

let instances = 0;

/** Starts a fresh worker.js and sends it `init`: its reply's value, or its error thrown. */
async function init(options, { isolated = false } = {}) {
  const messages = [];
  globalThis.self = {
    crossOriginIsolated: isolated,
    postMessage(message, transfer = []) {
      // As a real postMessage does: what is transferred is gone from the worker.
      const copy = structuredClone(message, { transfer });
      if (copy.type === 'store') globalThis.phononFake.log.push(`store ${copy.url}`);
      messages.push(copy);
    },
  };
  // A new module instance each time, with its own model and voices.
  await import(`../worker.js?instance=${instances++}`);
  const data = { type: 'init', id: 0, options: { lang: 'en', quant: 'q8', model: MODEL, preload: [], cache: true, conditions: [], ...options } };
  await self.onmessage({ data });
  const reply = messages.find((m) => m.id === 0 && (m.type === 'result' || m.type === 'error'));
  if (reply.type === 'error') throw new Error(reply.message);
  return reply.value;
}

const log = () => globalThis.phononFake.log.filter((e) => e !== 'wasm.init' && !e.startsWith('store'));
const warnings = () => globalThis.phononFake.warnings;

test('the CPU by default: no adapter asked for, and nothing tried on WebGPU', async () => {
  globalThis.phononFake.adapter = null;
  const r = await init({});
  assert.equal(r.device, 'cpu');
  assert.equal(r.deviceReason, 'default');
  assert.deepEqual(log(), ['Model.load cpu']);
});

test('WebGPU runs on the engine, and nothing loads into wasm', async () => {
  for (const device of ['webgpu', 'auto']) {
    globalThis.phononFake.log = [];
    const r = await init({ device, preload: ['a'] });
    assert.equal(r.device, 'webgpu');
    assert.equal(r.threads, 1);
    assert.deepEqual(log(), ['requestDevice', 'Preparer.new', 'GpuEngine.create', 'GpuEngine.addVoice']);
  }
  assert.deepEqual(warnings(), []);
});

test('a checkpoint the engine refuses: auto takes the CPU, webgpu names the reason', async () => {
  const model = { ...MODEL, config: 'mem:config-2steps' };
  const r = await init({ device: 'auto', model });
  assert.equal(r.device, 'cpu');
  assert.match(r.deviceReason, /does not run lsd_decode_steps 2/);
  assert.deepEqual(log(), ['Model.load cpu']);
  assert.equal(warnings().length, 1);

  globalThis.phononFake.log = [];
  await assert.rejects(init({ device: 'webgpu', model }), /does not run lsd_decode_steps 2/);
  assert.deepEqual(log(), []);
});

test('an engine that fails to start frees its device and its Preparer before the CPU loads', async () => {
  const cases = [
    [{ deviceFails: 'no device' }, ['requestDevice']],
    [{ preparerFails: 'bad tensors' }, ['requestDevice', 'Preparer.new', 'device.destroy']],
    [{ engineFails: 'the weights take more than one storage binding allows' }, ['requestDevice', 'Preparer.new', 'GpuEngine.create', 'Preparer.free', 'device.destroy']],
    [{ validation: 'bad binding' }, ['requestDevice', 'Preparer.new', 'GpuEngine.create', 'Preparer.free', 'device.destroy']],
  ];
  for (const [fails, engineLog] of cases) {
    const message = Object.values(fails)[0];
    Object.assign(globalThis.phononFake, { deviceFails: null, preparerFails: null, engineFails: null, validation: null }, fails);

    globalThis.phononFake.log = [];
    globalThis.phononFake.warnings = [];
    const r = await init({ device: 'auto' });
    assert.equal(r.device, 'cpu');
    assert.ok(r.deviceReason.startsWith('WebGPU failed to start: '), r.deviceReason);
    assert.ok(r.deviceReason.includes(message), r.deviceReason);
    assert.deepEqual(log(), [...engineLog, 'Model.load cpu']);
    assert.equal(warnings().length, 1);

    // Under 'webgpu' the error is the caller's, and the device is freed all the same.
    globalThis.phononFake.log = [];
    await assert.rejects(init({ device: 'webgpu' }), (e) => e.message.includes(message));
    assert.deepEqual(log(), engineLog);
  }
});

test('each file downloads once, and reaches the page only once the model is final', async () => {
  globalThis.phononFake.engineFails = 'pipeline failed';
  // The engine reads every file, then the CPU does: the fakes throw on a detached buffer.
  assert.equal((await init({ device: 'auto' })).device, 'cpu');
  assert.deepEqual(globalThis.phononFake.fetches.sort(), ['mem:config', 'mem:tokenizer', 'mem:weights']);
  const all = globalThis.phononFake.log;
  const lastLoad = all.findLastIndex((e) => e.startsWith('Model.load') || e === 'GpuEngine.create');
  const stores = all.flatMap((e, i) => (e.startsWith('store') ? [i] : []));
  assert.equal(stores.length, 3);
  assert.ok(stores.every((i) => i > lastLoad), all.join(', '));
});

test("a voice that fails to preload is the caller's error, and starts no fallback", async () => {
  for (const options of [{ device: 'auto' }, { device: 'webgpu' }, {}]) {
    globalThis.phononFake.log = [];
    globalThis.phononFake.warnings = [];
    await assert.rejects(init({ ...options, preload: ['a', 'bad'] }), /^Error: bad voice file$/);
    // One model, and voice `a` registered with it: no second model for the voices map to hold
    // indices of, and no device thrown away.
    const loads = log().filter((e) => e.startsWith('Model.load') || e === 'GpuEngine.create');
    assert.equal(loads.length, 1, log().join(', '));
    assert.equal(log().filter((e) => e === 'GpuEngine.addVoice' || e.startsWith('Model.add_voice')).length, 1);
    assert.ok(!log().includes('device.destroy'));
    assert.deepEqual(warnings(), []);
  }
});

test('webgpu with no adapter is an error; auto takes the CPU', async () => {
  globalThis.phononFake.adapter = null;
  await assert.rejects(init({ device: 'webgpu' }), /this browser offers no WebGPU adapter/);
  const r = await init({ device: 'auto' });
  assert.equal(r.device, 'cpu');
  assert.equal(r.deviceReason, 'this browser offers no WebGPU adapter');
});

test('wasmGlueUrl loads the single-threaded glue module from where it is served', async () => {
  globalThis.phononFake.log = [];
  const value = await init({ wasmGlueUrl: 'https://example.com/glue/phonon_tts.js' });
  assert.ok(globalThis.phononFake.log.includes('glue.wasm.init'), globalThis.phononFake.log.join(', '));
  assert.ok(!globalThis.phononFake.log.includes('wasm.init'), 'the bundled module was not used');
  assert.ok(globalThis.phononFake.log.includes('GlueModel.load cpu'), globalThis.phononFake.log.join(', '));
  assert.equal(value.device, 'cpu');
});

test('threadsWasmGlueUrl loads the threaded glue module from where it is served', async () => {
  globalThis.phononFake.log = [];
  const value = await init({ threadsWasmGlueUrl: 'https://example.com/glue/phonon_tts_threads.js' }, { isolated: true });
  assert.ok(globalThis.phononFake.log.includes('glue.wasm.init'), globalThis.phononFake.log.join(', '));
  assert.ok(globalThis.phononFake.log.includes('glue.initThreadPool 2'), globalThis.phononFake.log.join(', '));
  assert.ok(globalThis.phononFake.log.includes('glue.start_cpu_pool 2'), globalThis.phononFake.log.join(', '));
  assert.ok(!globalThis.phononFake.log.includes('wasm.init'), 'the bundled module was not used');
  assert.equal(value.device, 'cpu');
});
