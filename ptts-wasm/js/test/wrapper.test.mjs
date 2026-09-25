// The wrapper's own logic -- request queueing, cancellation, errors, caching, WAV encoding --
// against a fake worker and a fake network, so it runs under plain `node --test` with no
// browser and no model. The worker and the wasm are exercised by the demo page.

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
      return this.reply({ type: 'result', id, value: { sampleRate: 24000, features: {} } });
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

const { PhononTTS, encodeWav, concatPcm } = await import('../index.js');
const { fetchBytes } = await import('../fetch.js');

const MODEL = { weights: { q8: 'w' }, tokenizer: 't', voices: { alba: 'a', marius: 'm' } };
const load = () => PhononTTS.load({ lang: 'en', model: MODEL, workerUrl: 'worker.js' });

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
    /unknown rewrite rule\(s\) colours/,
  );
  assert.equal(FakeWorker.last?.init, undefined, 'no worker should have been started');

  await PhononTTS.load({ lang: 'en', rewrites: 'none', model: MODEL, workerUrl: 'worker.js' });
  assert.equal(FakeWorker.last.init.rewrites, 'none');

  // Left out, it stays undefined all the way to `Model::new`, whose own default is every rule.
  await load();
  assert.equal(FakeWorker.last.init.rewrites, undefined);
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

// ---- caching ----

function fakeNetwork(body) {
  const store = new Map();
  let fetches = 0;
  globalThis.caches = {
    async open() {
      return {
        match: async (url) => store.get(url)?.clone(),
        put: async (url, response) => void store.set(url, new Response(await response.arrayBuffer())),
      };
    },
  };
  globalThis.fetch = async () => {
    fetches++;
    return new Response(body, { headers: { 'content-length': String(body.length) } });
  };
  return { fetches: () => fetches };
}

test('fetchBytes downloads once, then serves from the cache', async () => {
  const body = new Uint8Array(3000).map((_, i) => i % 251);
  const net = fakeNetwork(body);
  const progress = [];
  const first = await fetchBytes('https://x/model', { onProgress: (p) => progress.push(p) });
  const second = await fetchBytes('https://x/model', { onProgress: (p) => progress.push(p) });
  assert.deepEqual(first, body);
  assert.deepEqual(second, body);
  assert.equal(net.fetches(), 1);
  assert.equal(progress.at(-1).cached, true);
  assert.equal(progress.find((p) => !p.cached).total, 3000);

  await fetchBytes('https://x/model', { cache: false });
  assert.equal(net.fetches(), 2, 'cache: false always downloads');
});

test('fetchBytes still works where the Cache API is missing', async () => {
  fakeNetwork(new Uint8Array([7, 8]));
  delete globalThis.caches;
  assert.deepEqual(await fetchBytes('https://x/y'), new Uint8Array([7, 8]));
});

test('fetchBytes reports HTTP errors', async () => {
  globalThis.fetch = async () => new Response('nope', { status: 404 });
  await assert.rejects(fetchBytes('https://x/missing', { cache: false }), /HTTP 404/);
});

// ---- wav ----

test('encodeWav writes a 16-bit mono header and clips', async () => {
  const wav = new DataView(await encodeWav(new Float32Array([0, 1, -1, 2]), 24000).arrayBuffer());
  const ascii = (o) => String.fromCharCode(...[0, 1, 2, 3].map((i) => wav.getUint8(o + i)));
  assert.equal(ascii(0), 'RIFF');
  assert.equal(ascii(8), 'WAVE');
  assert.equal(wav.getUint16(22, true), 1);
  assert.equal(wav.getUint32(24, true), 24000);
  assert.equal(wav.getUint32(40, true), 8);
  assert.deepEqual([0, 1, 2, 3].map((i) => wav.getInt16(44 + 2 * i, true)), [0, 32767, -32768, 32767]);
  assert.deepEqual(concatPcm([new Float32Array([1]), new Float32Array([2, 3])]), new Float32Array([1, 2, 3]));
});
