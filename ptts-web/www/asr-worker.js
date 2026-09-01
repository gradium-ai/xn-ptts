// The ASR model, in its own worker so it gets its own thread and its own WebGPU
// device, independent of the TTS worker.
import init, { load_asr, asr_frame, asr_reset, asr_frame_size, probe } from './ptts_web.js';
import { decodeSentencepieceModel, UnigramTokenizer } from './tokenizer.js';

let tokenizer = null;
let ready = false;
let frameSize = 1920;
let delayFrames = 0;
// Frames are queued rather than processed on arrival: the model runs slower than
// realtime, so the microphone outruns it and the backlog has to be explicit.
const queue = [];
// The in-flight drain, so `flush` can await the run that is already going rather
// than returning immediately and declaring the transcript final too early.
let draining = null;
let words = [];

const post = (type, data = {}) => self.postMessage({ type, ...data });
const status = message => post('status', { message });

async function fetchWithProgress(url, label) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${label}: ${resp.status} ${resp.statusText} (${url})`);
  const total = parseInt(resp.headers.get('content-length') || '0', 10);
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    post('progress', { label, received, total });
  }
  const out = new Uint8Array(received);
  let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function setup({ base, dtype, language }) {
  status('starting wasm…');
  await init();
  post('device', JSON.parse(await probe()));

  status('downloading asr tokenizer…');
  const tok = await fetchWithProgress(`${base}/tokenizer.model`, 'asr tokenizer');
  tokenizer = new UnigramTokenizer(decodeSentencepieceModel(tok));

  const config = await (await fetch(`${base}/config.json`)).text();
  status('downloading mimi encoder…');
  const mimi = await fetchWithProgress(`${base}/mimi.safetensors`, 'mimi');
  status('downloading asr lm…');
  const lm = await fetchWithProgress(`${base}/model.safetensors`, 'asr lm');

  status(`building the asr model on the GPU (${dtype})…`);
  const info = JSON.parse(await load_asr(mimi, lm, config, dtype, 0.0, language || null));
  frameSize = asr_frame_size();
  delayFrames = info.delay_frames;
  ready = true;
  post('ready', { info });
}

function drain() {
  if (!draining) draining = runQueue().finally(() => { draining = null; });
  return draining;
}

async function runQueue() {
  while (queue.length) {
    const pcm = queue.shift();
    const t0 = performance.now();
    let out;
    try {
      out = JSON.parse(await asr_frame(pcm));
    } catch (e) {
      post('error', { message: String(e && e.message || e) });
      break;
    }
    for (const w of out.words) {
      const text = tokenizer.decode(w.tokens);
      words.push(text);
      post('word', { text, start: w.start, backlog: queue.length });
    }
    post('frame_done', { ms: performance.now() - t0, backlog: queue.length });
    if (out.eos) post('eos', {});
  }
}

self.onmessage = async ev => {
  const { type, ...args } = ev.data;
  try {
    if (type === 'setup') {
      await setup(args);
    } else if (type === 'frame') {
      if (!ready) return;
      queue.push(args.pcm);
      drain();
    } else if (type === 'flush') {
      // The model runs `delay_frames` behind, so silence is fed in to push the
      // tail of the utterance out before the transcript is called final.
      if (!ready) return;
      for (let i = 0; i < delayFrames; i++) queue.push(new Float32Array(frameSize));
      await drain();
      post('final', { text: words.join('').trim() });
    } else if (type === 'reset') {
      queue.length = 0;
      words = [];
      if (ready) asr_reset();
      post('reset_done', {});
    }
  } catch (e) {
    post('error', { message: String(e && e.message || e) });
  }
};
