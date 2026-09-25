// Downloads, kept in the Cache API so a model is fetched once per browser rather than once
// per page load. The HTTP cache cannot be relied on for this: browsers cap the size of a
// single entry well below the ~150-240 MB a checkpoint weighs.

/** Bump to orphan everything cached by an older, incompatible version of this package. */
export const CACHE_NAME = 'phonon-tts-v1';

async function openCache() {
  // Absent outside secure contexts (plain http other than localhost), and it can throw
  // where storage is blocked. Either way the download still works, it just is not kept.
  if (typeof caches === 'undefined') return null;
  try {
    return await caches.open(CACHE_NAME);
  } catch {
    return null;
  }
}

/**
 * Fetch `url` as bytes, from the cache when it is there.
 *
 * @param {string} url
 * @param {{ cache?: boolean, onProgress?: (p: { loaded: number, total: number | null, cached: boolean }) => void }} [options]
 * @returns {Promise<Uint8Array>}
 */
export async function fetchBytes(url, { cache = true, onProgress } = {}) {
  const store = cache ? await openCache() : null;
  if (store) {
    const hit = await store.match(url).catch(() => undefined);
    if (hit) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      onProgress?.({ loaded: bytes.length, total: bytes.length, cached: true });
      return bytes;
    }
  }

  const response = await fetch(url);
  if (!response.ok) throw new Error(`failed to fetch ${url}: HTTP ${response.status}`);

  // Stored from a clone, streamed alongside the read below, rather than from the finished
  // buffer: `new Response(bytes)` would copy the whole checkpoint once more.
  const stored = store
    ? store.put(url, response.clone()).catch((e) => {
        // Usually the storage quota. Not fatal: the next load downloads again.
        console.warn(`[phonon-tts] could not cache ${url}: ${e}`);
      })
    : null;

  const length = Number(response.headers.get('content-length'));
  const total = Number.isFinite(length) && length > 0 ? length : null;
  const bytes = await readBody(response, total, (loaded) =>
    onProgress?.({ loaded, total, cached: false }),
  );
  await stored;
  return bytes;
}

async function readBody(response, total, onChunk) {
  if (!response.body) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  // Read straight into one buffer when the size is known, so a 240 MB download is not held
  // twice at the end.
  let buffer = new Uint8Array(total ?? 1 << 20);
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (loaded + value.length > buffer.length) {
      const grown = new Uint8Array(Math.max(buffer.length * 2, loaded + value.length));
      grown.set(buffer.subarray(0, loaded));
      buffer = grown;
    }
    buffer.set(value, loaded);
    loaded += value.length;
    onChunk(loaded);
  }
  return loaded === buffer.length ? buffer : buffer.slice(0, loaded);
}

/**
 * Delete every file this package has cached.
 *
 * @returns {Promise<boolean>} whether there was anything to delete.
 */
export async function clearCache() {
  if (typeof caches === 'undefined') return false;
  return caches.delete(CACHE_NAME);
}
