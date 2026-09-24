// The worker that owns the wasm model. `index.js` starts it and talks to it; nothing else
// should need to.
//
// Generation is synchronous inside wasm, one frame per `generation_step` call, so the loop
// below yields to the event loop between frames. That is what lets a `cancel` message land
// mid-utterance.

import init, { Model, cpu_features } from './wasm/phonon_tts.js';
import { fetchBytes } from './fetch.js';

let model = null;
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

async function handleInit(id, options) {
  settings = options;
  try {
    await init(options.wasmUrl ? { module_or_path: options.wasmUrl } : undefined);
  } catch (e) {
    if (e instanceof WebAssembly.CompileError) {
      throw new Error(
        `this browser cannot run phonon-tts: it needs WebAssembly SIMD and Relaxed SIMD (${e.message})`,
      );
    }
    throw e;
  }

  const { model: spec, quant, cache } = options;
  const weightsUrl = spec.weights[quant];
  if (!weightsUrl) throw new Error(`this model has no '${quant}' weights`);
  const progress = (file) => (p) => post({ type: 'progress', id, file, ...p });

  const [weights, tokenizer, config] = await Promise.all([
    fetchBytes(weightsUrl, { cache, onProgress: progress('weights') }),
    fetchBytes(spec.tokenizer, { cache, onProgress: progress('tokenizer') }),
    spec.config ? fetchBytes(spec.config, { cache, onProgress: progress('config') }) : null,
  ]);
  model = new Model(weights, tokenizer, config ?? undefined, quant, options.lang);

  for (const name of options.preload) await voiceIndex(name, progress(`voice:${name}`));
  return { sampleRate: model.sample_rate(), features: cpu_features() };
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
        const pcm = model.generation_step();
        if (pcm === undefined) break;
        const dt = performance.now() - s0;
        stepMsTotal += dt;
        stepMsMin = Math.min(stepMsMin, dt);
        stats.stepMs.max = Math.max(stats.stepMs.max, dt);
        stats.frames++;
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
