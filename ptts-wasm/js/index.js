// phonon-tts: on-device text-to-speech in the browser.
//
// `PhononTTS` is the public API. The model runs in a dedicated worker (`worker.js`), so
// generating never blocks the page; this file only posts requests to it and turns its
// replies into promises and async iterators.

import { DEFAULT_MODEL, resolveHubModel } from './hub.js';
import { concatPcm, encodeWav } from './wav.js';
import { storeBytes } from './fetch.js';

export { clearCache } from './fetch.js';
export { DEFAULT_MODEL } from './hub.js';
export { encodeWav, concatPcm } from './wav.js';

const LANGS = ['en', 'fr', 'de', 'es', 'pt', 'none'];
/** Rewrite rules the Rust side knows, beyond the whole-string values below. */
const RULES = [
  'numbers', 'currency', 'dashed-digits', 'emails', 'urls', 'abbreviations', 'elongations',
  'phones', 'times', 'dates',
];
/** Values `Rules`'s `FromStr` accepts only as the entire string, never inside a list. */
const WHOLE_RULES = ['default', 'all', 'none', 'off'];

/** The seed Rust and Python use when none is given, so the same request gives the same audio. */
const DEFAULT_SEED = 4242424242424242n;

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

  /** Samples per second of the audio this model produces (24000 for Phonon). */
  sampleRate;
  /** SIMD features the wasm module was built with, e.g. `{ simd128: true }`. */
  features;
  /** Where generation runs: `'webgpu'` or `'cpu'`. */
  device;
  /** Why `device` is what it is, e.g. `'default'` or `'requested'`. */
  deviceReason;
  /** CPU threads generation runs on, counting the worker that owns the model. */
  threads;
  /** Why `threads` is what it is, e.g. `'default'` or `'the page is not cross-origin isolated'`. */
  threadsReason;
  /** The languages the model was trained for, as listed in its config. */
  languages;

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
      conditions,
      quant = 'q8',
      model = DEFAULT_MODEL,
      voices,
      cache = true,
      onProgress,
      workerUrl,
      wasmUrl,
      threadsWasmUrl,
      simdWasmUrl,
      threadsSimdWasmUrl,
      threads = 'auto',
      device,
    } = options ?? {};
    // Required, as in every other frontend: the spoken forms of `@`, `+` and `=` differ per
    // language, so normalizing German text as English is worse than not normalizing at all.
    if (!LANGS.includes(lang)) {
      throw new TypeError(`lang is required: one of ${LANGS.map((l) => `'${l}'`).join(', ')}`);
    }
    if (quant !== 'f32' && quant !== 'q8') {
      throw new TypeError(`quant must be 'f32' or 'q8', got '${quant}'`);
    }
    if (device !== undefined && device !== 'auto' && device !== 'webgpu' && device !== 'cpu') {
      throw new TypeError(`device must be 'auto', 'webgpu' or 'cpu', got ${JSON.stringify(device)}`);
    }
    if (device === 'webgpu' && quant !== 'q8') {
      throw new TypeError("device 'webgpu' needs quant 'q8': f32 weights cannot be quantized on the GPU");
    }
    if (threads !== 'auto' && !(Number.isInteger(threads) && threads >= 1)) {
      throw new TypeError(`threads must be 'auto' or a positive integer, got ${JSON.stringify(threads)}`);
    }
    // Checked here rather than left to Rust: `load` is async and the error would otherwise
    // arrive after the weights had been downloaded. This mirrors `Rules`'s `FromStr`, which trims
    // and lowercases the whole string, takes `all`, `none` and `off` only on their own, and
    // otherwise reads a comma-separated list of rule names. Accepting anything it rejects
    // would put the error back after the download, which is what this check is here to avoid.
    if (rewrites !== undefined) {
      if (typeof rewrites !== 'string') throw new TypeError('rewrites must be a string');
      const value = rewrites.trim().toLowerCase();
      const unknown = WHOLE_RULES.includes(value)
        ? []
        : value.split(',').map((r) => r.trim()).filter((r) => !RULES.includes(r));
      if (unknown.length > 0) {
        throw new TypeError(
          `unknown rewrite rule(s) ${unknown.map((r) => `'${r}'`).join(', ')}: expected ` +
            `${WHOLE_RULES.map((r) => `'${r}'`).join(', ')}, or ${RULES.map((r) => `'${r}'`).join(', ')}`,
        );
      }
    }
    // Which names a checkpoint takes is only known once its config is loaded, so Rust checks
    // those; the shape is checked here, before the download.
    const isPlain = typeof conditions === 'object' && conditions !== null && !Array.isArray(conditions);
    if (conditions !== undefined && !isPlain) {
      throw new TypeError('conditions must be an object of name to value, e.g. { duration_delta: 0.5 }');
    }
    const conditionList = Object.entries(conditions ?? {}).map(([name, value]) => {
      if (typeof value !== 'string' && !Number.isFinite(value)) {
        throw new TypeError(`condition '${name}' must be a string or a finite number`);
      }
      return `${name}=${value}`;
    });
    // Checked last, so a bad option fails before a Hub request.
    const spec = typeof model === 'string' ? await resolveHubModel(model, { cache }) : model;
    if (typeof spec !== 'object' || spec === null || Array.isArray(spec)) {
      throw new TypeError(
        "model must say where the checkpoint's files are, e.g. " +
          "{ weights: { q8: '/model/model.q8.gguf' }, tokenizer: '/model/tokenizer.json', " +
          "config: '/model/config.json', voices: { Freya: '/model/voices/Freya.safetensors' } }, " +
          "or name a Hugging Face repo, 'hf://owner/name'. Leave it out for the default model.",
      );
    }
    if (!isUrl(spec.tokenizer)) {
      throw new TypeError("model.tokenizer is required: the URL of the checkpoint's tokenizer.json");
    }
    if (!isUrl(spec.config) || String(spec.config).length === 0) {
      throw new TypeError("model.config is required: the URL of the checkpoint's config.json");
    }
    // Checked before the worker starts, for the same reason as `rewrites` above.
    if (!isUrl(spec.weights?.[quant])) {
      throw new TypeError(`model has no '${quant}' weights: set model.weights.${quant}, or pick another quant`);
    }
    const defaultVoice = spec.defaultVoice ?? Object.keys(spec.voices ?? {})[0];
    const preload = voices ?? (defaultVoice ? [defaultVoice] : []);

    // `new URL(..., import.meta.url)` inline, not through a variable: it is the pattern Vite,
    // webpack 5, Parcel and esbuild recognise and rewrite when bundling the worker.
    const worker = workerUrl
      ? new Worker(workerUrl, { type: 'module' })
      : new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
    const tts = new PhononTTS(worker, spec, defaultVoice);
    try {
      const reply = await tts.#request(
        {
          type: 'init',
          options: {
            lang,
            rewrites,
            conditions: conditionList,
            quant,
            model: resolveModel(spec),
            preload,
            cache,
            wasmUrl: wasmUrl ? resolveUrl(wasmUrl) : undefined,
            threadsWasmUrl: threadsWasmUrl ? resolveUrl(threadsWasmUrl) : undefined,
            simdWasmUrl: simdWasmUrl ? resolveUrl(simdWasmUrl) : undefined,
            threadsSimdWasmUrl: threadsSimdWasmUrl ? resolveUrl(threadsSimdWasmUrl) : undefined,
            threads,
            device,
          },
        },
        { onProgress },
      );
      tts.sampleRate = reply.sampleRate;
      tts.features = reply.features;
      tts.device = reply.device;
      tts.languages = reply.languages;
      if (lang !== 'none' && tts.languages.length && !tts.languages.includes(lang)) {
        console.warn(
          `[phonon-tts] text is normalized as ${lang}, but the model was trained for ${tts.languages.join(', ')}`,
        );
      }
      tts.deviceReason = reply.deviceReason;
      tts.threads = reply.threads;
      tts.threadsReason = reply.threadsReason;
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
    worker.onmessage = ({ data }) => {
      // A file the worker downloaded, for the page to cache: see `storeBytes` for why.
      if (data.type === 'store') return void storeBytes(data.url, data.bytes);
      this.#pending.get(data.id)?.[data.type]?.(data);
    };
    worker.onerror = (e) => {
      e.preventDefault?.();
      // A dead worker never replies: fail what is pending and refuse what comes next, rather
      // than leaving later requests waiting forever.
      const error = new Error(`phonon-tts worker failed: ${e.message || 'could not start'}`);
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
    const { voice = this.#defaultVoice, temperature = 0.3, signal } = options;
    const seed = toSeed(options.seed ?? DEFAULT_SEED);
    const id = this.#nextId++;
    // Chunks the consumer has not taken yet, and the consumer waiting for the next one.
    const buffered = [];
    let waiting = null;
    let finished = false;
    let failure = null;
    let started = false;
    let cancelled = false;

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
      buffered.length = 0;
      if (finished || cancelled) return;
      cancelled = true;
      if (started) this.#worker.postMessage({ type: 'cancel', id });
      else finish(null, { cancelled: true });
    };
    if (signal?.aborted) cancel();
    else signal?.addEventListener('abort', cancel, { once: true });

    const handlers = {
      chunk: ({ pcm }) => {
        if (cancelled) return;
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

/** A seed as the model takes it: a `u64`, carried as a `bigint`. */
function toSeed(seed) {
  const value = typeof seed === 'bigint' ? seed : Number.isSafeInteger(seed) ? BigInt(seed) : null;
  if (value === null || value < 0n || value >= 1n << 64n) {
    // `String`, not a template literal, which throws on a symbol.
    const got = `${String(seed)} (${typeof seed})`;
    throw new RangeError(
      `seed must be a whole number from 0 to 2^64 - 1, as a number up to ${Number.MAX_SAFE_INTEGER} or a bigint; got ${got}`,
    );
  }
  return value;
}

/** What `resolveUrl` takes: a URL string, relative to the page or absolute, or a `URL`. */
function isUrl(url) {
  return typeof url === 'string' || url instanceof URL;
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
    config: resolveUrl(config),
    voices: map(voices ?? {}),
  };
}

function disposedError() {
  return new Error('this PhononTTS has been disposed');
}
