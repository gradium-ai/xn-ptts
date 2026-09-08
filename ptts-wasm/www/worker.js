import init, { Model, cpu_features } from './ptts_wasm.js';

const HF_BASE = 'https://huggingface.co/kyutai/pocket-tts-without-voice-cloning/resolve/main';
const HF_BASE_Q8 = 'https://huggingface.co/lmz/pocket-tts-without-voice-cloning-q8/resolve/main';
const TOKENIZER_URL = `${HF_BASE}/tokenizer.json`;

function modelUrl(quant) {
  if (quant === 'q8') return `${HF_BASE_Q8}/tts_b6369a24.gguf`;
  return `${HF_BASE}/tts_b6369a24.safetensors`;
}

function voiceUrl(name) {
  return `${HF_BASE}/embeddings_v2/${name}.safetensors`;
}

function post(type, data = {}, transferables = []) {
  self.postMessage({ type, ...data }, transferables);
}

// ---- Fetch with progress (posts to main thread) ----
async function fetchWithProgress(url, label) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (total > 0) {
      const pct = Math.round(received / total * 100);
      post('progress', {
        label,
        pct,
        detail: `${(received / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`,
      });
    } else {
      post('progress', { label, pct: -1, detail: `${(received / 1e6).toFixed(1)} MB` });
    }
  }
  post('progress_done');
  const buf = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buf.set(chunk, offset);
    offset += chunk.length;
  }
  return buf;
}

// ---- Worker state ----
const VOICE_NAMES = ['alba', 'marius', 'javert', 'fantine', 'cosette', 'eponine', 'azelma'];

// Start WASM compilation immediately so the optimizing compiler (TurboFan)
// finishes well before the first generation runs.
const wasmModulePromise = WebAssembly.compileStreaming(fetch('ptts_wasm_bg.wasm'));

let model = null;
let voiceIndexMap = {};

async function handleLoad(quant) {
  const wasmModule = await wasmModulePromise;
  await init(wasmModule);
  post('status', { message: 'WASM initialized. Downloading tokenizer and model...' });

  const tokenizerJson = await fetchWithProgress(TOKENIZER_URL, 'Tokenizer');
  const modelWeights = await fetchWithProgress(modelUrl(quant), 'Model weights');

  post('status', { message: `Initializing model (quant=${quant})...` });
  model = new Model(modelWeights, tokenizerJson, quant);

  for (const name of VOICE_NAMES) {
    post('status', { message: `Loading voice: ${name}...` });
    const voiceData = await fetchWithProgress(voiceUrl(name), `Voice: ${name}`);
    voiceIndexMap[name] = model.add_voice(voiceData);
  }

  const sampleRate = model.sample_rate();
  const features = cpu_features();
  post('loaded', { sampleRate, features });
}

async function handleGenerate(text, voiceName, temperature) {
  const voiceIndex = voiceIndexMap[voiceName];

  // Time the prompt step: `start_generation` prepares and tokenizes the text, then runs
  // `prompt_text` on the transformer state, which is the bulk of the prefill cost.
  const promptT0 = performance.now();
  const numTokens = model.start_generation(voiceIndex, text, temperature);
  const promptMs = performance.now() - promptT0;

  post('gen_start', { numTokens });

  let step = 0;
  let stepMsTotal = 0;
  let stepMsMin = Infinity;
  let stepMsMax = 0;
  while (true) {
    const t0 = performance.now();
    const chunk = model.generation_step();
    const dt = performance.now() - t0;
    if (!chunk) break;
    stepMsTotal += dt;
    if (dt < stepMsMin) stepMsMin = dt;
    if (dt > stepMsMax) stepMsMax = dt;
    post('chunk', { data: chunk, step }, [chunk.buffer]);
    step++;
  }

  const stepMsAvg = step > 0 ? stepMsTotal / step : 0;
  if (step === 0) stepMsMin = 0;
  post('done', {
    promptMs,
    numSteps: step,
    stepMsAvg,
    stepMsMin,
    stepMsMax,
  });
}

self.onmessage = async (e) => {
  const { type, ...data } = e.data;
  try {
    if (type === 'load') {
      await handleLoad(data.quant || 'f32');
    } else if (type === 'generate') {
      await handleGenerate(data.text, data.voiceName, data.temperature);
    }
  } catch (err) {
    post('error', { message: err.message });
    console.error(err);
  }
};
