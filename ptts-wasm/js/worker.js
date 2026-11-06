// The worker that owns the model. `index.js` starts it and talks to it; nothing else should
// need to.
//
// There are two engines behind one protocol. On the CPU the wasm `Model` holds the whole
// model, and each `generation_step` returns one frame; the loop yields to the event loop
// between steps, which is what lets a `cancel` message land mid-utterance. On WebGPU the
// engine in `gpu/` runs the model on the device and the wasm `Preparer` only prepares the
// text and the voices; its loop waits on GPU readbacks, which yields to the event loop too.

import { fetchBytes } from './fetch.js';
import { chooseDevice, isGguf, settleDevice } from './device.js';
import { chooseThreads } from './threads.js';
import { hasRelaxedSimd } from './simd.js';
import { GpuEngine } from './gpu/engine.js';
import { parseGguf, safetensorsOf } from './gpu/gguf.js';
import { chooseReductions, seed32, unsupportedConfig } from './gpu/support.js';

/** The CPU `Model`, or `{ gpu: true, preparer, engine }` on WebGPU. */
let model = null;
/** Samples per frame: a generation step returns a whole number of them. */
let frameSize = 0;
let settings = null;
/** Voice name -> index in `model`, or the pending load of that voice. */
const voices = new Map();
/** Ids of generations asked to stop. */
const cancelled = new Set();

const post = (message, transfer = []) => self.postMessage(message, transfer);

// A macrotask without the 4 ms clamp nested `setTimeout`s get. One frame is 80 ms of audio,
// so the clamp alone would cost several percent of real time.
const channel = new MessageChannel();
const yieldQueue = [];
channel.port1.onmessage = () => yieldQueue.shift()?.();
const yieldToEventLoop = () =>
  new Promise((resolve) => {
    yieldQueue.push(resolve);
    channel.port2.postMessage(null);
  });

/**
 * Each build comes as two modules with the same JS: one using WebAssembly Relaxed SIMD and
 * one without, for browsers that lack it. This picks the one the browser can compile, and
 * hands the JS its URL. The `new URL` calls name their file literally, so a bundler emits
 * the module beside the JS as it does the default one.
 */
const RELAXED_SIMD = hasRelaxedSimd();
function moduleFor(relaxedUrl, simdUrl, simdDefault) {
  if (RELAXED_SIMD) return relaxedUrl ? { module_or_path: relaxedUrl } : undefined;
  return { module_or_path: simdUrl ?? simdDefault };
}

/**
 * Load the wasm build to run on the CPU, and start its threads.
 *
 * There are two builds. `wasm-threads/` runs generation on several threads, but its memory
 * is shared, which a page can only create when it is cross-origin isolated; `wasm/` runs on
 * this thread alone and loads anywhere. The threaded one is tried when the page allows it
 * and more than one thread is wanted, and the single-threaded one otherwise or if that
 * fails, so a page that cannot have threads still speaks.
 *
 * Both `import()`s name their file literally, which is what lets a bundler find and emit
 * both builds.
 */
async function loadWasm(options) {
  const choice = chooseThreads({
    requested: options.threads,
    isolated: self.crossOriginIsolated === true,
    hardwareConcurrency: navigator.hardwareConcurrency,
  });
  let reason = choice.reason;
  if (choice.threads > 1) {
    try {
      const wasm = await import('./wasm-threads/phonon_tts.js');
      await wasm.default(
        moduleFor(
          options.threadsWasmUrl,
          options.threadsSimdWasmUrl,
          new URL('./wasm-threads/phonon_tts_simd_bg.wasm', import.meta.url),
        ),
      );
      const workers = choice.threads - 1;
      // rayon's Web Workers first, then xn's CPU pool on them; both before any model loads.
      await wasm.initThreadPool(workers);
      return { wasm, threads: wasm.start_cpu_pool(workers), reason };
    } catch (e) {
      // The threaded module stays loaded, with its shared memory and whatever Web Workers
      // it had started: wasm has no way to unload it. It is only reached when threads fail.
      reason = `threads failed to start: ${e instanceof Error ? e.message : e}`;
    }
  }
  return { wasm: await loadSingleThreaded(options), threads: 1, reason };
}

/**
 * The single-threaded build: the CPU build for pages that cannot have threads, and the one
 * the WebGPU engine prepares its text and voices with.
 */
async function loadSingleThreaded(options) {
  const wasm = await import('./wasm/phonon_tts.js');
  try {
    await wasm.default(
      moduleFor(options.wasmUrl, options.simdWasmUrl, new URL('./wasm/phonon_tts_simd_bg.wasm', import.meta.url)),
    );
  } catch (e) {
    if (e instanceof WebAssembly.CompileError) {
      throw new Error(
        `this browser cannot run phonon-tts: it needs WebAssembly SIMD (${e.message})`,
      );
    }
    throw e;
  }
  return wasm;
}

/** What WebGPU adapter the browser hands out, if any: the cheap check before loading weights. */
async function webGpuAdapter() {
  try {
    const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) return { hasWebGpu: false };
    // `info.isFallbackAdapter` in current browsers, `isFallbackAdapter` in older Chrome.
    return { hasWebGpu: true, adapter, fallbackAdapter: Boolean(adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter) };
  } catch {
    return { hasWebGpu: false };
  }
}

async function handleInit(id, options) {
  settings = options;
  const { model: spec, quant } = options;
  const wantsGpu = options.device === 'webgpu' || options.device === 'auto';
  const gpu = wantsGpu ? await webGpuAdapter() : { hasWebGpu: false };
  let choice = chooseDevice({ requested: options.device, quant, hasWebGpu: gpu.hasWebGpu, fallbackAdapter: gpu.fallbackAdapter });
  if (choice.device === 'webgpu' && !gpu.hasWebGpu) throw new Error('this browser offers no WebGPU adapter');

  const weightsUrl = spec.weights[quant];
  if (!weightsUrl) throw new Error(`this model has no '${quant}' weights`);
  const progress = (file) => (p) => post({ type: 'progress', id, file, ...p });
  // Each file downloads once, whichever engine ends up reading it.
  const pending = {};
  const file = (name, url) => (pending[name] ??= download(url, progress(name)));
  const weights = () => file('weights', weightsUrl);
  const tokenizer = () => file('tokenizer', spec.tokenizer);
  const config = () => file('config', spec.config);

  if (choice.device === 'webgpu') {
    // The config says whether the WebGPU engine runs this checkpoint: read it before the weights.
    const cfg = parseConfig((await config()).bytes);
    const settled = settleDevice({ requested: options.device, choice, unsupported: unsupportedConfig(cfg) });
    if ('error' in settled) throw new Error(settled.error);
    if (settled.warning) console.warn(`[phonon-tts] ${settled.warning}`);
    choice = settled;
  }
  if (choice.device === 'webgpu') {
    let info = null;
    try {
      info = await initGpu(options, gpu.adapter, { weights, tokenizer, config }, choice.reason);
    } catch (e) {
      if (options.device === 'webgpu') throw e;
      const why = `WebGPU failed to start: ${e instanceof Error ? e.message : e}`;
      console.warn(`[phonon-tts] ${why}; generating on the CPU instead`);
      choice = { device: 'cpu', reason: why };
      model = null;
    }
    if (info) {
      // Outside the try: a voice that fails to load is the caller's error on either device,
      // not a reason to run on the CPU.
      await preloadVoices(options, progress);
      return info;
    }
  }

  // The module first, so a browser that cannot compile it finds out before the download.
  const { wasm, threads, reason: threadsReason } = await loadWasm(options);
  const files = await Promise.all([weights(), tokenizer(), config()]);
  const [w, t, c] = files.map((f) => f.bytes);
  model = await wasm.Model.load(w, t, c, quant, options.lang, options.rewrites, options.conditions, 'cpu');
  frameSize = model.frame_size();
  // The model holds its own copy now: the page can have these to cache.
  for (const f of files) f.keep();

  await preloadVoices(options, progress);
  return {
    sampleRate: model.sample_rate(),
    features: wasm.cpu_features(),
    device: 'cpu',
    languages: model.languages(),
    deviceReason: choice.reason,
    threads,
    threadsReason,
  };
}

/** The voices `options.preload` names, once the model they register with is final. */
async function preloadVoices(options, progress) {
  for (const name of options.preload) await voiceIndex(name, progress(`voice:${name}`));
}

function parseConfig(bytes) {
  try {
    return JSON.parse(new TextDecoder().decode(bytes));
  } catch (e) {
    throw new Error(`cannot parse config.json: ${e instanceof Error ? e.message : e}`);
  }
}

/**
 * The WebGPU engine: the weights go from the GGUF to the device, and only the small tensors
 * the conditioners and the speaker projection need reach wasm, with the tokenizer and the
 * config, for the `Preparer`. When it fails, it frees what it made before throwing, so a CPU
 * fallback does not hold a second copy of the model beside it.
 */
async function initGpu(options, adapter, files, deviceReason) {
  const wasm = await loadSingleThreaded(options);
  const reductions = chooseReductions(adapter);
  const device = await adapter.requestDevice({ requiredFeatures: reductions.subgroups ? ['subgroups'] : [] });
  device.lost.then((info) => {
    if (info.reason !== 'destroyed') console.error(`[phonon-tts] the WebGPU device was lost: ${info.message}`);
  });
  let preparer = null;
  try {
    const all = await Promise.all([files.weights(), files.tokenizer(), files.config()]);
    const [weights, tokenizer, config] = all.map((f) => f.bytes);
    if (!isGguf(weights)) throw new Error('WebGPU needs q8 weights in a GGUF file');
    const prefix = 'flow_lm.condition_provider.conditioners.';
    const tensors = safetensorsOf(parseGguf(weights), [prefix], [`${prefix}transcript_in_segment.`]);
    preparer = new wasm.Preparer(tokenizer, config, tensors, options.lang, options.rewrites, options.conditions);
    device.pushErrorScope('validation');
    const engine = await GpuEngine.create(device, {
      gguf: weights,
      config: parseConfig(config),
      conditions: new Float32Array(preparer.conditions()),
      subgroups: reductions.subgroups,
    });
    const error = await device.popErrorScope();
    if (error) throw new Error(`WebGPU rejected the engine: ${error.message}`);
    const info = {
      sampleRate: preparer.sample_rate(),
      features: wasm.cpu_features(),
      device: 'webgpu',
      languages: preparer.languages(),
      deviceReason,
      threads: 1,
      threadsReason: 'generation runs on the GPU',
    };
    model = { gpu: true, preparer, engine, device };
    frameSize = preparer.frame_size();
    // The weights are on the device now: the page can have the files to cache. Last, since
    // they are transferred: a CPU fallback after this would find them gone.
    for (const f of all) f.keep();
    return info;
  } catch (e) {
    preparer?.free();
    device.destroy();
    throw e;
  }
}

/**
 * Downloads `url`, or reads it from the cache, without writing the cache: `keep()` then hands a
 * download to the page, which caches it (see `storeBytes` in fetch.js for why the worker does
 * not). Call it only once the bytes are no longer needed here: they are transferred, not
 * copied, so they are gone from this side afterwards. wasm copies what it is given, and the
 * WebGPU engine what it uploads, so that is as soon as the model or the voice has been loaded.
 */
async function download(url, onProgress) {
  let cached = false;
  const bytes = await fetchBytes(url, {
    cache: settings.cache,
    write: false,
    onProgress: (p) => {
      cached = p.cached;
      onProgress?.(p);
    },
  });
  const keep = () => {
    if (!settings.cache || cached) return;
    const buffer = bytes.byteLength === bytes.buffer.byteLength ? bytes.buffer : bytes.slice().buffer;
    post({ type: 'store', url, bytes: buffer }, [buffer]);
  };
  return { bytes, keep };
}

/** The index of voice `name`, fetching and registering it on first use. */
function voiceIndex(name, onProgress) {
  let entry = voices.get(name);
  if (entry === undefined) {
    const url = settings.model.voices[name];
    if (!url) {
      const known = [...new Set([...Object.keys(settings.model.voices), ...voices.keys()])];
      return Promise.reject(new Error(`unknown voice '${name}'; known: ${known.join(', ')}`));
    }
    entry = download(url, onProgress)
      .then(async ({ bytes, keep }) => {
        const index = await registerVoice(bytes);
        keep();
        return index;
      })
      .catch((e) => {
        voices.delete(name); // let a later call retry
        throw e;
      });
    voices.set(name, entry);
  }
  return Promise.resolve(entry);
}

async function handleAddVoice(name, source) {
  const { bytes, keep } =
    typeof source === 'string' ? await download(source) : { bytes: new Uint8Array(source), keep() {} };
  const index = registerVoice(bytes);
  voices.set(name, index);
  try {
    await index;
    keep();
  } catch (e) {
    voices.delete(name); // let a later call retry
    throw e;
  }
}

/** Registers a voice and waits until the device has run its prompt. */
async function registerVoice(bytes) {
  if (model.gpu) return model.engine.addVoice(model.preparer.voice(bytes));
  const index = model.add_voice(bytes);
  await model.flush();
  return index;
}

async function handleGenerate(id, request) {
  return model.gpu ? generateGpu(id, request) : generateCpu(id, request);
}

function newStats(chunks) {
  return { chunks, tokens: 0, frames: 0, samples: 0, promptMs: 0, stepMs: { avg: 0, min: 0, max: 0 }, firstAudioMs: null, totalMs: 0, cancelled: false };
}

/** Per-frame step times: each piece of audio's time since the previous one, over its frames. */
function stepTimer(stats) {
  let total = 0;
  let min = Infinity;
  return {
    add(ms, frames) {
      const dt = ms / frames;
      total += ms;
      min = Math.min(min, dt);
      stats.stepMs.max = Math.max(stats.stepMs.max, dt);
    },
    finish() {
      if (stats.frames > 0) {
        stats.stepMs.avg = total / stats.frames;
        stats.stepMs.min = min;
      }
    },
  };
}

async function generateGpu(id, { text, voice, temperature, seed }) {
  const index = await voiceIndex(voice);
  const t0 = performance.now();
  const chunks = model.preparer.plan(text);
  const stats = newStats(chunks.length);
  const timer = stepTimer(stats);
  const base = seed32(seed);
  try {
    for (let i = 0; i < chunks.length; i++) {
      if (cancelled.has(id)) {
        stats.cancelled = true;
        break;
      }
      const chunk = chunks[i];
      stats.tokens += chunk.tokens.length;
      let last = performance.now();
      const result = await model.engine.speakChunk(chunk, {
        voice: index,
        // Each chunk's noise from its own seed, so a seed still fixes the whole text.
        seed: (base ^ Math.imul(i + 1, 0x9e3779b9)) >>> 0,
        temperature,
        cancelled: () => cancelled.has(id),
        onAudio: (pcm) => {
          const now = performance.now();
          const frames = Math.max(1, Math.round(pcm.length / frameSize));
          timer.add(now - last, frames);
          last = now;
          stats.frames += frames;
          stats.samples += pcm.length;
          stats.firstAudioMs ??= now - t0;
          post({ type: 'chunk', id, pcm }, [pcm.buffer]);
        },
      });
      if (result.cancelled) {
        stats.cancelled = true;
        break;
      }
    }
  } finally {
    cancelled.delete(id);
  }
  timer.finish();
  stats.totalMs = performance.now() - t0;
  return stats;
}

async function generateCpu(id, { text, voice, temperature, seed }) {
  const index = await voiceIndex(voice);
  const t0 = performance.now();
  const chunks = model.start_generation(index, text, temperature, seed);
  const stats = newStats(chunks);
  const timer = stepTimer(stats);
  try {
    outer: for (;;) {
      // Checked here as well as in the frame loop: a cancel that lands on the yield after a
      // chunk's last frame would otherwise still prompt the next chunk before it took effect.
      if (cancelled.has(id)) {
        stats.cancelled = true;
        break;
      }
      const p0 = performance.now();
      const tokens = model.next_chunk();
      if (tokens === undefined) break;
      stats.tokens += tokens;
      stats.promptMs += performance.now() - p0;

      for (;;) {
        if (cancelled.has(id)) {
          stats.cancelled = true;
          break outer;
        }
        const s0 = performance.now();
        const pcm = await model.generation_step();
        if (pcm === undefined) break;
        const frames = Math.max(1, Math.round(pcm.length / frameSize));
        timer.add(performance.now() - s0, frames);
        stats.frames += frames;
        stats.samples += pcm.length;
        stats.firstAudioMs ??= performance.now() - t0;
        post({ type: 'chunk', id, pcm }, [pcm.buffer]);
        await yieldToEventLoop();
      }
    }
  } finally {
    try {
      model.stop_generation();
    } catch {
      // After a wasm trap every call on `model` throws; keep the original error.
    }
    cancelled.delete(id);
  }
  timer.finish();
  stats.totalMs = performance.now() - t0;
  return stats;
}

self.onmessage = async ({ data }) => {
  const { type, id } = data;
  if (type === 'cancel') {
    cancelled.add(id);
    return;
  }
  try {
    let value;
    if (type === 'init') value = await handleInit(id, data.options);
    else if (type === 'add_voice') value = await handleAddVoice(data.name, data.source);
    else if (type === 'generate') value = await handleGenerate(id, data);
    else throw new Error(`unknown message type '${type}'`);
    post({ type: 'result', id, value });
  } catch (e) {
    post({ type: 'error', id, message: e instanceof Error ? e.message : String(e) });
  }
};
