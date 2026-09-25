// phonon-tts: on-device text-to-speech in the browser.
//
// `PhononTTS` is the public API. The model runs in a dedicated worker (`worker.js`), so
// generating never blocks the page; this file only posts requests to it and turns its
// replies into promises and async iterators.

import { DEFAULT_MODEL } from './models.js';
import { concatPcm, encodeWav } from './wav.js';

export { DEFAULT_MODEL } from './models.js';
export { clearCache } from './fetch.js';
export { encodeWav, concatPcm } from './wav.js';

const LANGS = ['en', 'fr', 'de', 'es', 'pt', 'none'];
/** Rewrite rules the Rust side knows, beyond `'all'` and `'none'`. */
const RULES = ['numbers'];

export class PhononTTS {
  #worker;
  #nextId = 0;
  /** Request id -> handlers for its replies. */
  #pending = new Map();
  /** Tail of the generation queue: the worker runs one generation at a time. */
  #queue = Promise.resolve();
  #voices;
  #defaultVoice;
  #disposed = false;
  /** Why the instance stopped, when it was not `dispose()`: the worker's crash. */
  #failure = null;

  /** Samples per second of the audio this model produces (24000 for Pocket TTS). */
  sampleRate;
  /** SIMD features the wasm module was built with, e.g. `{ simd128: true }`. */
  features;

  /**
   * Download (or read from cache) a checkpoint and start it in a worker.
   *
   * @param {import('./index.js').LoadOptions} options
   * @returns {Promise<PhononTTS>}
   */
  static async load(options) {
    const {
      lang,
      rewrites,
      quant = 'q8',
      model = DEFAULT_MODEL,
      voices,
      cache = true,
      onProgress,
      workerUrl,
      wasmUrl,
    } = options ?? {};
    // Required, as in every other frontend: the spoken forms of `@`, `+` and `=` differ per
    // language, so normalizing German text as English is worse than not normalizing at all.
    if (!LANGS.includes(lang)) {
      throw new TypeError(`lang is required: one of ${LANGS.map((l) => `'${l}'`).join(', ')}`);
    }
    if (quant !== 'f32' && quant !== 'q8') {
      throw new TypeError(`quant must be 'f32' or 'q8', got '${quant}'`);
    }
    // Checked here rather than left to Rust: `load` is async and the error would otherwise
    // arrive after the weights had been downloaded.
    if (rewrites !== undefined) {
      const unknown = rewrites
        .split(',')
        .map((r) => r.trim())
        .filter((r) => r !== 'all' && r !== 'none' && !RULES.includes(r));
      if (unknown.length > 0) {
        throw new TypeError(
          `unknown rewrite rule(s) ${unknown.join(', ')}: expected 'all', 'none', or ` +
            RULES.map((r) => `'${r}'`).join(', '),
        );
      }
    }
    const defaultVoice = model.defaultVoice ?? Object.keys(model.voices ?? {})[0];
    const preload = voices ?? (defaultVoice ? [defaultVoice] : []);

    // `new URL(..., import.meta.url)` inline, not through a variable: it is the pattern Vite,
    // webpack 5, Parcel and esbuild recognise and rewrite when bundling the worker.
    const worker = workerUrl
      ? new Worker(workerUrl, { type: 'module' })
      : new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    const tts = new PhononTTS(worker, model, defaultVoice);
    try {
      const { sampleRate, features } = await tts.#request(
        {
          type: 'init',
          options: {
            lang,
            rewrites,
            quant,
            model: resolveModel(model),
            preload,
            cache,
            wasmUrl: wasmUrl ? resolveUrl(wasmUrl) : undefined,
          },
        },
        { onProgress },
      );
      tts.sampleRate = sampleRate;
      tts.features = features;
      return tts;
    } catch (e) {
      tts.dispose();
      throw e;
    }
  }

  /** @private Use `PhononTTS.load`. */
  constructor(worker, model, defaultVoice) {
    this.#worker = worker;
    this.#voices = new Set(Object.keys(model.voices ?? {}));
    this.#defaultVoice = defaultVoice;
    worker.onmessage = ({ data }) => this.#pending.get(data.id)?.[data.type]?.(data);
    worker.onerror = (e) => {
      e.preventDefault?.();
      // A dead worker never replies: fail what is pending and refuse what comes next, rather
      // than leaving later requests waiting forever.
      const error = new Error(`phonon-tts worker failed: ${e.message ?? 'could not start'}`);
      this.#disposed = true;
      this.#failure = error;
      worker.terminate();
      this.#failAll(error);
    };
  }

  /** Names of the voices this model can speak with: bundled ones and any added. */
  get voices() {
    return [...this.#voices];
  }

  /**
   * Register a voice under `name`, from a URL or from the bytes of a voice `.safetensors`
   * file: a precomputed embedding (`emb`) or the KV-cache format of `embeddings_v2/`.
   *
   * @param {string} name
   * @param {string | URL | ArrayBuffer | Uint8Array | Blob} source
   */
  addVoice(name, source) {
    // Queued with the generations: a `stream` for this voice called right after, without an
    // `await`, then waits for it, and adding a voice never stalls audio already being made.
    const added = this.#queue.then(() => this.#addVoiceNow(name, source));
    this.#queue = added.catch(() => {});
    return added;
  }

  async #addVoiceNow(name, source) {
    let payload;
    const transfer = [];
    if (typeof source === 'string' || source instanceof URL) {
      payload = resolveUrl(source);
    } else {
      const bytes = source instanceof Blob ? await source.arrayBuffer() : source;
      // Copied rather than transferred, so the caller's buffer stays usable.
      payload = bytes instanceof ArrayBuffer ? bytes.slice(0) : bytes.slice().buffer;
      transfer.push(payload);
    }
    await this.#request({ type: 'add_voice', name, source: payload }, { transfer });
    this.#voices.add(name);
  }

  /**
   * Speak `text`, yielding audio as it is generated: mono `Float32Array`s at `sampleRate`,
   * 80 ms each. Long text is split at sentence boundaries and spoken chunk by chunk.
   *
   * @param {string} text
   * @param {import('./index.js').SpeechOptions} [options]
   * @returns {import('./index.js').SpeechStream}
   */
  stream(text, options = {}) {
    const { voice = this.#defaultVoice, temperature = 0.3, seed = 42, signal } = options;
    const id = this.#nextId++;
    // Chunks the consumer has not taken yet, and the consumer waiting for the next one.
    const buffered = [];
    let waiting = null;
    let finished = false;
    let failure = null;
    let started = false;

    let resolveDone, rejectDone;
    const done = new Promise((res, rej) => ((resolveDone = res), (rejectDone = rej)));
    // Nobody has to await `done`; a failure still reaches them through the iterator.
    done.catch(() => {});

    const wake = () => {
      const w = waiting;
      waiting = null;
      w?.();
    };
    const finish = (err, stats) => {
      if (finished) return;
      finished = true;
      failure = err;
      signal?.removeEventListener('abort', cancel);
      err ? rejectDone(err) : resolveDone(stats);
      wake();
    };
    const cancel = () => {
      if (finished) return;
      if (started) this.#worker.postMessage({ type: 'cancel', id });
      else finish(null, { cancelled: true });
    };
    if (signal?.aborted) cancel();
    else signal?.addEventListener('abort', cancel, { once: true });

    const handlers = {
      chunk: ({ pcm }) => {
        buffered.push(pcm);
        wake();
      },
      result: ({ value }) => {
        this.#pending.delete(id);
        finish(null, value);
      },
      error: ({ message }) => {
        this.#pending.delete(id);
        finish(new Error(message));
      },
    };

    const run = () =>
      new Promise((settled) => {
        done.then(settled, settled);
        if (finished || this.#disposed) return finish(this.#disposed ? this.#endError() : null, { cancelled: true });
        started = true;
        this.#pending.set(id, handlers);
        this.#worker.postMessage({ type: 'generate', id, text, voice, temperature, seed });
      });
    this.#queue = this.#queue.then(run, run);

    const stream = {
      sampleRate: this.sampleRate,
      done,
      cancel,
      async *[Symbol.asyncIterator]() {
        try {
          for (;;) {
            if (buffered.length) yield buffered.shift();
            else if (finished) break;
            else await new Promise((w) => (waiting = w));
          }
          if (failure) throw failure;
        } finally {
          // A consumer that breaks out of its loop is done listening.
          cancel();
        }
      },
    };
    return stream;
  }

  /**
   * Speak `text` and return the whole waveform once it is done.
   *
   * @param {string} text
   * @param {import('./index.js').SpeechOptions} [options]
   * @returns {Promise<Float32Array>}
   */
  async synth(text, options) {
    const chunks = [];
    for await (const pcm of this.stream(text, options)) chunks.push(pcm);
    return concatPcm(chunks);
  }

  /**
   * Speak `text` and return it as a 16-bit mono WAV file.
   *
   * @param {string} text
   * @param {import('./index.js').SpeechOptions} [options]
   * @returns {Promise<Blob>}
   */
  async synthWav(text, options) {
    return encodeWav(await this.synth(text, options), this.sampleRate);
  }

  /** Stop the worker and free the model. Every pending request fails. */
  dispose() {
    if (this.#disposed) return;
    this.#disposed = true;
    this.#worker.terminate();
    this.#failAll(disposedError());
  }

  #endError() {
    return this.#failure ?? disposedError();
  }

  #failAll(error) {
    for (const handlers of this.#pending.values()) handlers.error({ message: error.message });
    this.#pending.clear();
  }

  #request(message, { onProgress, transfer = [] } = {}) {
    if (this.#disposed) return Promise.reject(this.#endError());
    const id = this.#nextId++;
    return new Promise((resolve, reject) => {
      this.#pending.set(id, {
        progress: ({ file, loaded, total, cached }) => onProgress?.({ file, loaded, total, cached }),
        result: ({ value }) => {
          this.#pending.delete(id);
          resolve(value);
        },
        error: ({ message }) => {
          this.#pending.delete(id);
          reject(new Error(message));
        },
      });
      this.#worker.postMessage({ ...message, id }, transfer);
    });
  }
}

/**
 * An absolute URL for `url`. The worker resolves relative URLs against its own script, which
 * lives inside the package, so anything relative has to be made absolute against the page.
 */
function resolveUrl(url) {
  return globalThis.location ? new URL(url, globalThis.location.href).href : String(url);
}

function resolveModel({ weights, tokenizer, config, voices }) {
  const map = (o) => Object.fromEntries(Object.entries(o).map(([k, u]) => [k, resolveUrl(u)]));
  return {
    weights: map(weights),
    tokenizer: resolveUrl(tokenizer),
    config: config ? resolveUrl(config) : null,
    voices: map(voices ?? {}),
  };
}

function disposedError() {
  return new Error('this PhononTTS has been disposed');
}
