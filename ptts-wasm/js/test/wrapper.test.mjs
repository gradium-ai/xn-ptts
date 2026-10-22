// The wrapper's own logic -- request queueing, cancellation, errors -- against a fake worker,
// so it runs under plain `node --test` with no browser and no model. The worker and the wasm
// are exercised by the demo page.

import { test } from 'node:test';
import assert from 'node:assert/strict';

// ---- a fake of the worker protocol in worker.js ----

class FakeWorker {
  static last = null;
  /** Frames each `generate` produces, and what it does after them. */
  static script = { frames: 3, fail: null };

  constructor() {
    FakeWorker.last = this;
    this.cancelled = new Set();
    this.generations = [];
    this.terminated = false;
    this.log = [];
  }
  reply(data) {
    queueMicrotask(() => !this.terminated && this.onmessage?.({ data }));
  }
  postMessage(msg) {
    const { type, id } = msg;
    this.log.push(type);
    if (type === 'init') {
      this.init = msg.options;
      return this.reply({
        type: 'result',
        id,
        value: {
          sampleRate: 24000,
          features: {},
          device: 'cpu',
          deviceReason: 'this browser offers no WebGPU adapter',
          threads: 4,
          threadsReason: 'default',
          languages: ['en', 'fr'],
        },
      });
    }
    if (type === 'add_voice') return this.reply({ type: 'result', id });
    if (type === 'cancel') return this.cancelled.add(id);
    if (type === 'generate') {
      this.generations.push(msg);
      this.run(msg);
    }
  }
  async run({ id }) {
    const { frames, fail } = FakeWorker.script;
    let sent = 0;
    for (; sent < frames; sent++) {
      await new Promise((r) => setTimeout(r, 1));
      if (this.cancelled.has(id)) break;
      this.reply({ type: 'chunk', id, pcm: new Float32Array(4).fill(sent) });
    }
    await new Promise((r) => setTimeout(r, 1));
    if (fail) this.reply({ type: 'error', id, message: fail });
    else this.reply({ type: 'result', id, value: { frames: sent, cancelled: this.cancelled.has(id) } });
  }
  terminate() {
    this.terminated = true;
  }
}
globalThis.Worker = FakeWorker;

const { PhononTTS } = await import('../index.js');

const MODEL = { weights: { q8: 'w' }, tokenizer: 't', config: 'c', voices: { alba: 'a', marius: 'm' } };
const load = () => PhononTTS.load({ lang: 'en', model: MODEL, workerUrl: 'worker.js' });

test('device is checked before the worker starts, handed to it, and reported back', async () => {
  FakeWorker.last = null;
  for (const device of ['gpu', '', null, 1]) {
    await assert.rejects(PhononTTS.load({ lang: 'en', model: MODEL, device }), /device must be/);
  }
  // f32 weights cannot go to the GPU, so asking for both is refused up front.
  const f32 = { ...MODEL, weights: { f32: 'w' } };
  await assert.rejects(PhononTTS.load({ lang: 'en', model: f32, quant: 'f32', device: 'webgpu' }), /needs quant 'q8'/);
  assert.equal(FakeWorker.last, null);
  const tts = await PhononTTS.load({ lang: 'en', model: MODEL, device: 'webgpu', workerUrl: 'worker.js' });
  assert.equal(FakeWorker.last.init.device, 'webgpu');
  // What the worker says it got, not what was asked for.
  assert.equal(tts.device, 'cpu');
  assert.equal(tts.deviceReason, 'this browser offers no WebGPU adapter');
  assert.deepEqual(tts.languages, ['en', 'fr']);
  tts.dispose();
  // Left out, it stays out, so the worker can tell the CPU default from a request for it.
  await PhononTTS.load({ lang: 'en', model: MODEL, workerUrl: 'worker.js' });
  assert.equal(FakeWorker.last.init.device, undefined);
});

test('a language the model was not trained for is warned about', async (t) => {
  const warn = t.mock.method(console, 'warn', () => {});
  for (const lang of ['en', 'fr', 'none']) {
    (await PhononTTS.load({ lang, model: MODEL, workerUrl: 'worker.js' })).dispose();
  }
  assert.equal(warn.mock.callCount(), 0);
  (await PhononTTS.load({ lang: 'de', model: MODEL, workerUrl: 'worker.js' })).dispose();
  assert.deepEqual(warn.mock.calls.map((c) => c.arguments[0]), [
    '[phonon-tts] text is normalized as de, but the model was trained for en, fr',
  ]);
});

test('threads is checked before the worker starts, handed to it, and reported back', async () => {
  FakeWorker.last = null;
  for (const threads of [0, -1, 2.5, '4', null]) {
    await assert.rejects(PhononTTS.load({ lang: 'en', model: MODEL, threads }), /threads must be/);
  }
  assert.equal(FakeWorker.last, null);
  const tts = await PhononTTS.load({ lang: 'en', model: MODEL, threads: 2, workerUrl: 'worker.js' });
  assert.equal(FakeWorker.last.init.threads, 2);
  // What the worker says it got, not what was asked for.
  assert.equal(tts.threads, 4);
  assert.equal(tts.threadsReason, 'default');
  tts.dispose();
  await PhononTTS.load({ lang: 'en', model: MODEL, workerUrl: 'worker.js' });
  assert.equal(FakeWorker.last.init.threads, 'auto');
});

test('a model needs its files, with weights for the quant being loaded', async () => {
  FakeWorker.last = null;
  await assert.rejects(PhononTTS.load({ lang: 'en', model: null }), /model must say where/);
  await assert.rejects(PhononTTS.load({ lang: 'en', model: [] }), /model must say where/);
  await assert.rejects(PhononTTS.load({ lang: 'en', model: { weights: { q8: 'w' } } }), /model\.tokenizer is required/);
  await assert.rejects(PhononTTS.load({ lang: 'en', model: MODEL, quant: 'f32' }), /no 'f32' weights/);
  // All of them fail before a worker exists, so nothing has been downloaded.
  assert.equal(FakeWorker.last, null);
});

test('an hf:// model is resolved to URLs at its commit, after the other options are checked', async () => {
  const sha = 'abc123';
  const requests = [];
  globalThis.fetch = async (url) => {
    requests.push(url);
    const files = ['config.json', 'tokenizer.json', 'model.q8.gguf', 'voices/anna.safetensors'];
    return Response.json({ sha, siblings: files.map((rfilename) => ({ rfilename })) });
  };
  FakeWorker.last = null;
  await assert.rejects(PhononTTS.load({ lang: 'en', model: 'hf://acme/speech', rewrites: 'nope' }), /unknown rewrite/);
  await assert.rejects(PhononTTS.load({ lang: 'en', model: '/model/' }), /hf:\/\/owner\/name/);
  assert.deepEqual(requests, []);
  assert.equal(FakeWorker.last, null);

  const tts = await PhononTTS.load({ lang: 'en', model: 'hf://acme/speech@v1', workerUrl: 'worker.js' });
  const base = `https://huggingface.co/acme/speech/resolve/${sha}`;
  assert.deepEqual(requests, ['https://huggingface.co/api/models/acme/speech/revision/v1']);
  assert.deepEqual(FakeWorker.last.init.model, {
    weights: { q8: `${base}/model.q8.gguf`, f32: `${base}/model.q8.gguf` },
    tokenizer: `${base}/tokenizer.json`,
    config: `${base}/config.json`,
    voices: { anna: `${base}/voices/anna.safetensors` },
  });
  assert.deepEqual(FakeWorker.last.init.preload, ['anna']);
  assert.deepEqual(tts.voices, ['anna']);
  tts.dispose();
});

test('without a model, the default one is loaded from the Hub', async () => {
  const { DEFAULT_MODEL } = await import('../index.js');
  const [repo, revision] = DEFAULT_MODEL.slice('hf://'.length).split('@');
  const requests = [];
  globalThis.fetch = async (url) => {
    requests.push(url);
    const files = ['config.json', 'tokenizer.json', 'model.q8.gguf', 'embeddings/alba.safetensors'];
    return Response.json({ sha: revision, siblings: files.map((rfilename) => ({ rfilename })) });
  };
  const tts = await PhononTTS.load({ lang: 'en', workerUrl: 'worker.js' });
  assert.deepEqual(requests, [`https://huggingface.co/api/models/${repo}/revision/${revision}`]);
  assert.equal(FakeWorker.last.init.model.config, `https://huggingface.co/${repo}/resolve/${revision}/config.json`);
  tts.dispose();
});

test('model files may be URL objects, as workerUrl and wasmUrl may', async () => {
  const model = {
    weights: { q8: new URL('https://example.com/m/model.q8.gguf') },
    tokenizer: new URL('https://example.com/m/tokenizer.json'),
    config: new URL('https://example.com/m/config.json'),
    voices: { v: new URL('https://example.com/m/v.safetensors') },
  };
  await PhononTTS.load({ lang: 'en', model, workerUrl: 'worker.js' });
  assert.equal(FakeWorker.last.init.model.tokenizer, 'https://example.com/m/tokenizer.json');
  assert.equal(FakeWorker.last.init.model.weights.q8, 'https://example.com/m/model.q8.gguf');
});

test('every module URL reaches the worker resolved, the SIMD-only ones too', async () => {
  const urls = {
    wasmUrl: 'https://example.com/w/phonon_tts_bg.wasm',
    threadsWasmUrl: 'https://example.com/t/phonon_tts_bg.wasm',
    simdWasmUrl: new URL('https://example.com/w/phonon_tts_simd_bg.wasm'),
    threadsSimdWasmUrl: 'https://example.com/t/phonon_tts_simd_bg.wasm',
  };
  await PhononTTS.load({ lang: 'en', model: MODEL, workerUrl: 'worker.js', ...urls });
  for (const [key, value] of Object.entries(urls)) assert.equal(FakeWorker.last.init[key], String(value));
});

test('lang is required', async () => {
  await assert.rejects(PhononTTS.load({ model: MODEL }), /lang is required/);
  await assert.rejects(PhononTTS.load({ lang: 'xx', model: MODEL }), /lang is required/);
});

test('rewrites is checked before anything is downloaded, then handed to the worker', async () => {
  FakeWorker.last = null;
  // Rejected here rather than in Rust: `load` would otherwise fetch the weights first and
  // report the bad rule only once they had arrived.
  await assert.rejects(
    PhononTTS.load({ lang: 'en', rewrites: 'numbers,colours', model: MODEL }),
    /unknown rewrite rule\(s\) 'colours'/,
  );
  assert.equal(FakeWorker.last?.init, undefined, 'no worker should have been started');

  await PhononTTS.load({ lang: 'en', rewrites: 'none', model: MODEL, workerUrl: 'worker.js' });
  assert.equal(FakeWorker.last.init.rewrites, 'none');

  // Left out, it stays undefined all the way to `Model::new`, whose own default is `default`.
  await load();
  assert.equal(FakeWorker.last.init.rewrites, undefined);
});

test('the rewrites check accepts exactly what the Rust Rules parser accepts', async () => {
  // `Rules`'s `FromStr` trims and lowercases, and takes `off` as a synonym for `none`.
  for (const value of ['off', 'ALL', ' none ', ' Default', 'Numbers', 'numbers', 'emails, URLS', 'dashed-digits,dates']) {
    await PhononTTS.load({ lang: 'en', rewrites: value, model: MODEL, workerUrl: 'worker.js' });
    assert.equal(FakeWorker.last.init.rewrites, value, 'passed through as written');
  }
  // `default`, `all` and `none` are whole-string values in Rust, so a list containing one is an error
  // there. Catching it here is the whole point: otherwise it surfaces after the download.
  for (const value of ['all,numbers', 'default,emails', 'none,numbers', '', 'numbers,colours']) {
    await assert.rejects(
      PhononTTS.load({ lang: 'en', rewrites: value, model: MODEL }),
      /unknown rewrite rule/,
    );
  }
  await assert.rejects(
    PhononTTS.load({ lang: 'en', rewrites: ['numbers'], model: MODEL }),
    /rewrites must be a string/,
  );
});

test('every rule in the Rust RULES table is accepted', async () => {
  const { readFile } = await import('node:fs/promises');
  const rust = await readFile(new URL('../../../ptts/src/preprocess/rewrite.rs', import.meta.url), 'utf8');
  const names = [...rust.matchAll(/Rule \{ name: "([^"]+)"/g)].map((m) => m[1]);
  assert.ok(names.length > 0, 'the RULES table was found');
  for (const name of names) {
    await PhononTTS.load({ lang: 'en', rewrites: name, model: MODEL, workerUrl: 'worker.js' });
    assert.equal(FakeWorker.last.init.rewrites, name);
  }
});

test('conditions are handed to the worker as NAME=VALUE, their shape checked first', async () => {
  FakeWorker.last = null;
  for (const conditions of [['padding_bonus=0.5'], 'padding_bonus=0.5', null, { padding_bonus: NaN }, { n: true }]) {
    await assert.rejects(PhononTTS.load({ lang: 'en', conditions, model: MODEL }), /condition/);
  }
  assert.equal(FakeWorker.last, null, 'no worker should have been started');

  const conditions = { padding_bonus: 0.5, num_speakers: '2' };
  await PhononTTS.load({ lang: 'en', conditions, model: MODEL, workerUrl: 'worker.js' });
  assert.deepEqual(FakeWorker.last.init.conditions, ['padding_bonus=0.5', 'num_speakers=2']);
  await load();
  assert.deepEqual(FakeWorker.last.init.conditions, []);
});

test('stream yields every frame in order, then its stats', async () => {
  FakeWorker.script = { frames: 3, fail: null };
  const tts = await load();
  assert.deepEqual(tts.voices, ['alba', 'marius']);
  const stream = tts.stream('Hello.');
  const seen = [];
  for await (const pcm of stream) seen.push(pcm[0]);
  assert.deepEqual(seen, [0, 1, 2]);
  assert.equal((await stream.done).frames, 3);
  assert.equal(FakeWorker.last.generations[0].voice, 'alba', 'the default voice');
});

test('the seed defaults to the one Rust and Python use, and goes out as a bigint', async () => {
  FakeWorker.script = { frames: 1, fail: null };
  const tts = await load();
  for (const [seed, sent] of [[undefined, 4242424242424242n], [7, 7n], [2n ** 63n, 2n ** 63n]]) {
    await tts.synth('Hi.', { seed });
    assert.equal(FakeWorker.last.generations.at(-1).seed, sent);
  }
  for (const seed of [-1, 1.5, 2 ** 60, '7', 2n ** 64n, {}, Symbol('seed')]) {
    assert.throws(() => tts.stream('Hi.', { seed }), RangeError, String(seed));
  }
});

test('synth concatenates, and requests run one at a time in order', async () => {
  FakeWorker.script = { frames: 2, fail: null };
  const tts = await load();
  const [a, b] = await Promise.all([tts.synth('One.'), tts.synth('Two.', { voice: 'marius' })]);
  assert.equal(a.length, 8);
  assert.equal(b.length, 8);
  assert.deepEqual(
    FakeWorker.last.generations.map((g) => [g.text, g.voice]),
    [['One.', 'alba'], ['Two.', 'marius']],
  );
});

test('breaking out of the loop cancels the generation', async () => {
  FakeWorker.script = { frames: 50, fail: null };
  const tts = await load();
  const stream = tts.stream('Long text.');
  for await (const _ of stream) break;
  const stats = await stream.done;
  assert.equal(stats.cancelled, true);
  assert.ok(stats.frames < 50);
});

test('an aborted signal stops a queued request before it starts', async () => {
  FakeWorker.script = { frames: 3, fail: null };
  const tts = await load();
  const first = tts.stream('First.');
  const controller = new AbortController();
  const second = tts.stream('Second.', { signal: controller.signal });
  controller.abort();
  assert.equal((await second.done).cancelled, true);
  for await (const _ of first);
  assert.equal(FakeWorker.last.generations.length, 1, 'the cancelled request never reached the worker');
});

test('a worker error rejects both the iterator and done, after the frames it sent', async () => {
  FakeWorker.script = { frames: 2, fail: 'boom' };
  const tts = await load();
  const stream = tts.stream('Hi.');
  const seen = [];
  await assert.rejects(async () => {
    for await (const pcm of stream) seen.push(pcm);
  }, /boom/);
  assert.equal(seen.length, 2);
  await assert.rejects(stream.done, /boom/);

  // The queue keeps going after a failure.
  FakeWorker.script = { frames: 1, fail: null };
  assert.equal((await tts.synth('Again.')).length, 4);
});

test('dispose fails pending and later requests', async () => {
  FakeWorker.script = { frames: 50, fail: null };
  const tts = await load();
  const stream = tts.stream('Long.');
  tts.dispose();
  await assert.rejects(stream.done, /disposed/);
  await assert.rejects(tts.synth('After.'), /disposed/);
});

test('addVoice copies bytes rather than detaching the caller\'s buffer', async () => {
  const tts = await load();
  const bytes = new Uint8Array([1, 2, 3]);
  await tts.addVoice('mine', bytes);
  assert.equal(bytes.length, 3);
  assert.ok(tts.voices.includes('mine'));
});

test('a stream right after addVoice waits for the voice', async () => {
  FakeWorker.script = { frames: 1, fail: null };
  const tts = await load();
  const added = tts.addVoice('mine', new Blob([new Uint8Array([1])]));
  const pcm = await tts.synth('Hi.', { voice: 'mine' });
  await added;
  assert.equal(pcm.length, 4);
  assert.deepEqual(FakeWorker.last.log.slice(-2), ['add_voice', 'generate']);
});

test('after a worker crash, later requests fail with the crash', async () => {
  const tts = await load();
  FakeWorker.last.onerror({ message: 'out of memory' });
  await assert.rejects(tts.synth('After.'), /out of memory/);
  await assert.rejects(tts.addVoice('x', 'https://x/v'), /out of memory/);
});


test('config is required before a worker or download starts', async () => {
  FakeWorker.last = null;
  for (const config of [undefined, null, '', 1]) {
    await assert.rejects(PhononTTS.load({ lang: 'en', model: { ...MODEL, config } }), /model.config is required/);
  }
  assert.equal(FakeWorker.last, null);
});

test('cancel discards unread audio and the model can speak again', async () => {
  FakeWorker.script = { frames: 20, fail: null };
  const tts = await load();
  const worker = FakeWorker.last;
  let resolveBuffered;
  const buffered = new Promise((resolve) => { resolveBuffered = resolve; });
  const reply = worker.reply.bind(worker);
  let chunks = 0;
  worker.reply = (data) => {
    reply(data);
    if (data.type === 'chunk' && ++chunks === 3) queueMicrotask(resolveBuffered);
  };
  const speech = tts.stream('A longer sentence.');
  await buffered;
  speech.cancel();
  const iterator = speech[Symbol.asyncIterator]();
  assert.equal((await iterator.next()).done, true, 'cancelled audio must not reach playback');
  assert.equal((await speech.done).cancelled, true);
  FakeWorker.script = { frames: 2, fail: null };
  assert.equal((await tts.synth('Speak again.')).length, 8);
  tts.dispose();
});

test('cancel also discards audio still buffered after generation completed', async () => {
  FakeWorker.script = { frames: 3, fail: null };
  const tts = await load();
  const speech = tts.stream('Hello.');
  await speech.done;
  speech.cancel();
  assert.equal((await speech[Symbol.asyncIterator]().next()).done, true);
  tts.dispose();
});

test('a file the worker hands back is cached by the page, not taken for a reply', async () => {
  const stored = new Map();
  globalThis.caches = {
    async open() {
      return { put: async (url, res) => void stored.set(url, new Uint8Array(await res.arrayBuffer())) };
    },
  };
  try {
    await load();
    FakeWorker.last.onmessage({ data: { type: 'store', url: 'https://x/w', bytes: new Uint8Array([4, 5]).buffer } });
    await new Promise((r) => setTimeout(r, 0));
    assert.deepEqual(stored.get('https://x/w'), new Uint8Array([4, 5]));
  } finally {
    delete globalThis.caches;
  }
});
