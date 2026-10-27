// The download cache's own logic, against a fake Cache API and a fake network, so it runs
// under plain `node --test` with no browser and no model.

import { test } from 'node:test';
import assert from 'node:assert/strict';

const { fetchBytes, storeBytes } = await import('../fetch.js');

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

test('write: false reads the cache but leaves a download out of it, for storeBytes to keep', async () => {
  const body = new Uint8Array([1, 2, 3]);
  const net = fakeNetwork(body);
  assert.deepEqual(await fetchBytes('https://x/w', { write: false }), body);
  await fetchBytes('https://x/w', { write: false });
  assert.equal(net.fetches(), 2, 'nothing was cached');

  assert.equal(await storeBytes('https://x/w', body.slice().buffer), true);
  const progress = [];
  assert.deepEqual(await fetchBytes('https://x/w', { write: false, onProgress: (p) => progress.push(p) }), body);
  assert.equal(net.fetches(), 2, 'served from what storeBytes kept');
  assert.equal(progress.at(-1).cached, true);
});

test('storeBytes reports a failed write rather than throwing', async () => {
  globalThis.caches = {
    async open() {
      return {
        put: async () => {
          throw new Error('QuotaExceededError');
        },
      };
    },
  };
  const warn = console.warn;
  console.warn = () => {};
  try {
    assert.equal(await storeBytes('https://x/w', new Uint8Array([1])), false);
  } finally {
    console.warn = warn;
  }
  delete globalThis.caches;
  assert.equal(await storeBytes('https://x/w', new Uint8Array([1])), false, 'no Cache API');
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


test('fetchBytes grows its buffer when the server sends no content-length', async () => {
  // The path a chunked response with no length takes: start at 1 MiB, double, then trim.
  const chunk = new Uint8Array(700_000).map((_, i) => i % 253);
  globalThis.fetch = async () =>
    new Response(
      new ReadableStream({
        start(c) {
          for (let i = 0; i < 3; i++) c.enqueue(chunk);
          c.close();
        },
      }),
    );
  const progress = [];
  const bytes = await fetchBytes('https://x/big', { cache: false, onProgress: (p) => progress.push(p) });
  assert.equal(bytes.length, 3 * chunk.length, 'trimmed back to what arrived');
  assert.deepEqual(bytes.subarray(2 * chunk.length), chunk, 'the last chunk survived the growth');
  assert.equal(progress.at(-1).total, null, 'no total to report');
});
