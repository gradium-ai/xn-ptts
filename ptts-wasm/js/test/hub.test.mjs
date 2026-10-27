// Resolving `hf://` model sources, against a fake Hub API, so it runs under plain
// `node --test` with no network.

import { test } from 'node:test';
import assert from 'node:assert/strict';

const { DEFAULT_MODEL, parseHubModel, resolveHubModel } = await import('../hub.js');

const SHA = '0123456789abcdef0123456789abcdef01234567';
const BASE = `https://huggingface.co/acme/speech/resolve/${SHA}`;

/** A Hub that answers every API request with `files` at `SHA`, recording what was asked. */
function fakeHub(files, { status = 200, headers = {}, extra = {} } = {}) {
  const requests = [];
  globalThis.fetch = async (url) => {
    requests.push(String(url));
    const body = status === 200 ? { sha: SHA, siblings: files.map((rfilename) => ({ rfilename })), ...extra } : { error: 'x' };
    return new Response(JSON.stringify(body), { status, headers });
  };
  return requests;
}

const CHECKPOINT = ['config.json', 'tokenizer.json', 'model.q8.gguf', 'model.safetensors', 'README.md'];

test('parseHubModel reads the repo and an optional revision', () => {
  assert.deepEqual(parseHubModel('hf://acme/speech'), { repo: 'acme/speech', revision: 'main' });
  assert.deepEqual(parseHubModel('hf://acme/speech@v1.2'), { repo: 'acme/speech', revision: 'v1.2' });
  assert.deepEqual(parseHubModel(`hf://acme/my_model.v2@${SHA}`), { repo: 'acme/my_model.v2', revision: SHA });
  assert.deepEqual(parseHubModel('hf://acme/speech@refs/pr/3'), { repo: 'acme/speech', revision: 'refs/pr/3' });
});

test('parseHubModel rejects anything else, saying what it takes', () => {
  for (const bad of [
    '',
    '/model/',
    'https://huggingface.co/acme/speech',
    'acme/speech',
    'hf://speech',
    'hf://acme/speech/config.json',
    'hf://acme/speech@',
    'hf://acme/speech/',
    'hf://acme/ speech',
  ]) {
    assert.throws(() => parseHubModel(bad), /hf:\/\/owner\/name/, bad);
  }
});

test('a bad string fails before any request', async () => {
  const requests = fakeHub(CHECKPOINT);
  await assert.rejects(resolveHubModel('https://example.com/model/'), /cannot list a folder URL/);
  assert.equal(requests.length, 0);
});

test('one API request, then every file pinned to the commit it resolved to', async () => {
  const requests = fakeHub(CHECKPOINT);
  const spec = await resolveHubModel('hf://acme/speech');
  assert.deepEqual(requests, ['https://huggingface.co/api/models/acme/speech/revision/main']);
  assert.deepEqual(spec, {
    weights: { q8: `${BASE}/model.q8.gguf`, f32: `${BASE}/model.safetensors` },
    tokenizer: `${BASE}/tokenizer.json`,
    config: `${BASE}/config.json`,
    voices: {},
  });
});

test('the revision is sent escaped, as one path segment', async () => {
  const requests = fakeHub(CHECKPOINT);
  await resolveHubModel('hf://acme/speech@refs/pr/3');
  assert.deepEqual(requests, ['https://huggingface.co/api/models/acme/speech/revision/refs%2Fpr%2F3']);
});

test('weights fall back to the other format, as the native loader does', async () => {
  fakeHub(['config.json', 'tokenizer.json', 'model.safetensors']);
  const f32Only = await resolveHubModel('hf://acme/speech');
  assert.deepEqual(f32Only.weights, { q8: `${BASE}/model.safetensors`, f32: `${BASE}/model.safetensors` });
  fakeHub(['config.json', 'tokenizer.json', 'model.q8.gguf']);
  const q8Only = await resolveHubModel('hf://acme/speech');
  assert.deepEqual(q8Only.weights, { q8: `${BASE}/model.q8.gguf`, f32: `${BASE}/model.q8.gguf` });
});

test('voices follow the native layout: voices/ over embeddings/, then default-voice', async () => {
  fakeHub([
    ...CHECKPOINT,
    'embeddings/b.safetensors',
    'voices/b.safetensors',
    'embeddings/a.safetensors',
    'embeddings/notes.txt',
    'embeddings_v2/c.safetensors',
    'voices/sub/d.safetensors',
    'voices/with space.safetensors',
    'default-voice.safetensors',
  ]);
  const spec = await resolveHubModel('hf://acme/speech');
  assert.deepEqual(spec.voices, {
    a: `${BASE}/embeddings/a.safetensors`,
    b: `${BASE}/voices/b.safetensors`,
    default: `${BASE}/default-voice.safetensors`,
    'with space': `${BASE}/voices/with%20space.safetensors`,
  });
  assert.deepEqual(Object.keys(spec.voices), ['a', 'b', 'default', 'with space']);
  assert.equal(spec.defaultVoice, 'default');
});

test('a default voice in voices/ wins over default-voice.safetensors', async () => {
  fakeHub([...CHECKPOINT, 'voices/default.safetensors', 'default-voice.safetensors']);
  const spec = await resolveHubModel('hf://acme/speech');
  assert.deepEqual(spec.voices, { default: `${BASE}/voices/default.safetensors` });
});

test('without a default voice, the first by name is the default', async () => {
  fakeHub([...CHECKPOINT, 'voices/zoe.safetensors', 'voices/anna.safetensors']);
  const spec = await resolveHubModel('hf://acme/speech');
  assert.equal(spec.defaultVoice, 'anna');
});

test('missing files are named', async () => {
  fakeHub(['model.q8.gguf']);
  await assert.rejects(resolveHubModel('hf://acme/speech'), /has no config\.json, tokenizer\.json$/);
  fakeHub(['config.json', 'tokenizer.json', 'model.bin']);
  await assert.rejects(resolveHubModel('hf://acme/speech'), /has no weights \(model\.q8\.gguf or model\.safetensors\)/);
});

test('a missing, private or gated repo says the browser loads public repos only', async () => {
  for (const status of [401, 403, 404]) {
    fakeHub([], { status });
    await assert.rejects(resolveHubModel('hf://acme/speech'), /missing, private or gated.*public repos/);
  }
  // A gated repo's listing is public; its files are not.
  fakeHub(CHECKPOINT, { extra: { gated: 'manual' } });
  await assert.rejects(resolveHubModel('hf://acme/speech'), /missing, private or gated/);
});

test('an unknown revision is named as such', async () => {
  fakeHub([], { status: 404, headers: { 'x-error-code': 'RevisionNotFound' } });
  await assert.rejects(resolveHubModel('hf://acme/speech@nope'), /acme\/speech has no revision 'nope'/);
});

test('other failures say what was being looked up', async () => {
  fakeHub([], { status: 500 });
  await assert.rejects(resolveHubModel('hf://acme/speech'), /cannot look up hf:\/\/acme\/speech: HTTP 500/);
  globalThis.fetch = async () => {
    throw new TypeError('Failed to fetch');
  };
  await assert.rejects(resolveHubModel('hf://acme/speech'), /cannot reach the Hugging Face Hub.*Failed to fetch/);
});

test('offline, the last answer from the Hub is used, so cached files still load', async () => {
  const stored = new Map();
  globalThis.caches = {
    open: async () => ({
      match: async (url) => stored.get(url)?.clone(),
      put: async (url, response) => void stored.set(url, response),
    }),
  };
  try {
    fakeHub(CHECKPOINT);
    const online = await resolveHubModel('hf://acme/speech');
    globalThis.fetch = async () => {
      throw new TypeError('Failed to fetch');
    };
    assert.deepEqual(await resolveHubModel('hf://acme/speech'), online);
    for (const status of [503, 429]) {
      globalThis.fetch = async () => new Response('', { status });
      assert.deepEqual(await resolveHubModel('hf://acme/speech'), online, `HTTP ${status}`);
    }
    globalThis.fetch = async () => {
      throw new TypeError('Failed to fetch');
    };
    await assert.rejects(resolveHubModel('hf://acme/speech', { cache: false }), /cannot reach/);
    await assert.rejects(resolveHubModel('hf://acme/other'), /cannot reach/);
  } finally {
    delete globalThis.caches;
  }
});

test('an answer that is not JSON names what was being looked up', async () => {
  globalThis.fetch = async () => new Response('<html>sign in to the network</html>', { status: 200 });
  await assert.rejects(resolveHubModel('hf://acme/speech'), /hf:\/\/acme\/speech: the Hub's answer is not JSON/);
});

test('the default model is the one the native frontends use, pinned to a commit', async () => {
  const { readFile } = await import('node:fs/promises');
  const rust = await readFile(new URL('../../../ptts/src/checkpoint.rs', import.meta.url), 'utf8');
  const native = /pub const DEFAULT_MODEL: &str =\s*"([^"]+)";/.exec(rust)?.[1];
  assert.equal(DEFAULT_MODEL, native);
  assert.match(parseHubModel(DEFAULT_MODEL).revision, /^[0-9a-f]{40}$/);
});
