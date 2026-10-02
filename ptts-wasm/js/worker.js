// The worker that owns the wasm model. `index.js` starts it and talks to it; nothing else
// should need to.
//
// The model runs on WebGPU or on the CPU, through the same `Model` and the same loop: each
// `generation_step` returns a few frames, one on the CPU and several on WebGPU, and the loop
// below yields to the event loop between steps. That is what lets a `cancel` message land
// mid-utterance.

import { fetchBytes } from './fetch.js';
import { chooseDevice, isGguf } from './device.js';
import { chooseThreads } from './threads.js';

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
      await wasm.default(options.threadsWasmUrl ? { module_or_path: options.threadsWasmUrl } : undefined);
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
 * WebGPU runs in, since the GPU needs no CPU threads.
 */
async function loadSingleThreaded(options) {
  const wasm = await import('./wasm/phonon_tts.js');
  try {
    await wasm.default(options.wasmUrl ? { module_or_path: options.wasmUrl } : undefined);
  } catch (e) {
    if (e instanceof WebAssembly.CompileError) {
      throw new Error(
        `this browser cannot run phonon-tts: it needs WebAssembly SIMD and Relaxed SIMD (${e.message})`,
      );
    }
    throw e;
  }
  return wasm;
}

/** What WebGPU adapter the browser hands out, if any: the cheap check before loading weights. */
async function webGpuAdapter() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) return { hasWebGpu: false };
    // `info.isFallbackAdapter` in current browsers, `isFallbackAdapter` in older Chrome.
    return { hasWebGpu: true, fallbackAdapter: Boolean(adapter.info?.isFallbackAdapter ?? adapter.isFallbackAdapter) };
  } catch {
    return { hasWebGpu: false };
  }
}

async function handleInit(id, options) {
  settings = options;
  const { model: spec, quant, cache } = options;
  const adapter = options.device === 'cpu' ? { hasWebGpu: false } : await webGpuAdapter();
  const choice = chooseDevice({ requested: options.device, quant, ...adapter });
  if (choice.device === 'webgpu' && !adapter.hasWebGpu) throw new Error('this browser offers no WebGPU adapter');
  // The module first, so a browser that cannot compile it finds out before the download.
  let { wasm, threads, reason: threadsReason } =
    choice.device === 'webgpu'
      ? { wasm: await loadSingleThreaded(options), threads: 1, reason: 'generation runs on the GPU' }
      : await loadWasm(options);

  const weightsUrl = spec.weights[quant];
  if (!weightsUrl) throw new Error(`this model has no '${quant}' weights`);
  const progress = (file) => (p) => post({ type: 'progress', id, file, ...p });

  const [weights, tokenizer, config] = await Promise.all([
    fetchBytes(weightsUrl, { cache, onProgress: progress('weights') }),
    fetchBytes(spec.tokenizer, { cache, onProgress: progress('tokenizer') }),
    spec.config ? fetchBytes(spec.config, { cache, onProgress: progress('config') }) : null,
  ]);
  const load = (device) =>
    wasm.Model.load(weights, tokenizer, config ?? undefined, quant, options.lang, options.rewrites, device);

  let deviceReason = choice.reason;
  if (choice.device === 'webgpu') {
    try {
      if (!isGguf(weights)) throw new Error('WebGPU needs q8 weights in a GGUF file');
      model = await load('webgpu');
    } catch (e) {
      if (options.device === 'webgpu') throw e;
      deviceReason = `WebGPU failed to start: ${e instanceof Error ? e.message : e}`;
      // The CPU runs in the build already loaded, single threaded. Loading the threaded one
      // would copy the weights into a second module's memory, and wasm memory never shrinks,
      // so a device whose GPU failed would carry both copies.
      threadsReason = 'WebGPU failed, and the CPU stays on the single-threaded build already loaded';
    }
  }
  model ??= await load('cpu');
  frameSize = model.frame_size();

  for (const name of options.preload) await voiceIndex(name, progress(`voice:${name}`));
  return {
    sampleRate: model.sample_rate(),
    features: wasm.cpu_features(),
    device: model.device(),
    deviceReason,
    threads,
    threadsReason,
  };
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
    entry = fetchBytes(url, { cache: settings.cache, onProgress })
      .then((bytes) => model.add_voice(bytes))
      .catch((e) => {
        voices.delete(name); // let a later call retry
        throw e;
      });
    voices.set(name, entry);
  }
  return Promise.resolve(entry);
}

async function handleAddVoice(name, source) {
  const bytes = typeof source === 'string' ? await fetchBytes(source, { cache: settings.cache }) : new Uint8Array(source);
  voices.set(name, model.add_voice(bytes));
}

async function handleGenerate(id, { text, voice, temperature, seed }) {
  const index = await voiceIndex(voice);
  const t0 = performance.now();
  const chunks = model.start_generation(index, text, temperature, seed >>> 0);

  const stats = { chunks, tokens: 0, frames: 0, samples: 0, promptMs: 0, stepMs: { avg: 0, min: 0, max: 0 }, firstAudioMs: null, totalMs: 0, cancelled: false };
  let stepMsTotal = 0;
  let stepMsMin = Infinity;
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
        // One frame on the CPU, several on WebGPU, which reads a step back in one go.
        const pcm = await model.generation_step();
        if (pcm === undefined) break;
        const frames = Math.max(1, Math.round(pcm.length / frameSize));
        const dt = (performance.now() - s0) / frames;
        stepMsTotal += dt * frames;
        stepMsMin = Math.min(stepMsMin, dt);
        stats.stepMs.max = Math.max(stats.stepMs.max, dt);
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
  if (stats.frames > 0) {
    stats.stepMs.avg = stepMsTotal / stats.frames;
    stats.stepMs.min = stepMsMin;
  }
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
