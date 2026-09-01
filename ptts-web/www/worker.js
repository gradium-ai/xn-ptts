// Everything model-related happens here, off the UI thread: fetching weights,
// building the WebGPU device, and the generation loop. Audio frames are posted
// to the page as they are produced.
import init, {
  probe, load_model, add_voice, generate, prepare_text,
  threads_info, max_frames_for_tokens,
} from './ptts_web.js';
import { decodeSentencepieceModel, UnigramTokenizer } from './tokenizer.js';

let tokenizer = null;
let ready = false;

const post = (type, data = {}, transfer = []) => self.postMessage({ type, ...data }, transfer);
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

async function setup({ base, dtype, temperature, voices, configUrl, weightsUrl }) {
  status('starting wasm…');
  await init();

  post('threads', JSON.parse(threads_info()));

  status('asking the browser for a WebGPU adapter…');
  const dev = JSON.parse(await probe());
  post('device', dev);

  let configJson = null;
  if (configUrl) {
    try {
      const r = await fetch(configUrl);
      if (r.ok) configJson = await r.text();
    } catch (e) {
      // No config.json is fine: the built-in v202601 config is then used.
    }
  }

  status('downloading tokenizer…');
  const tokBytes = await fetchWithProgress(`${base}/tokenizer.model`, 'tokenizer');
  tokenizer = new UnigramTokenizer(decodeSentencepieceModel(tokBytes));

  // The container is the caller's choice because the dtype dictates it: a q8 run
  // needs pre-quantized blocks, which only the gguf has.
  const wUrl = weightsUrl || `${base}/model.safetensors`;
  status(`downloading weights from ${wUrl}…`);
  const weights = await fetchWithProgress(wUrl, 'weights');
  const magic = String.fromCharCode(...weights.slice(0, 4));
  post('weights', { url: wUrl, bytes: weights.length, magic });

  status(`building the model on the GPU (${dtype})…`);
  const info = JSON.parse(await load_model(weights, configJson, dtype, temperature));

  for (const v of voices) {
    try {
      const bytes = await fetchWithProgress(v.url, `voice ${v.name}`);
      add_voice(v.name, bytes);
      post('voice', { name: v.name });
    } catch (e) {
      post('warn', { message: `voice ${v.name}: ${e.message}` });
    }
  }

  ready = true;
  post('ready', { info });
}

async function run({ text, voice, seed }) {
  if (!ready) throw new Error('model is not loaded yet');
  const [prepared, framesAfterEos] = prepare_text(text);
  const ids = Array.from(tokenizer.encode(prepared));
  // The budget, not the outcome: eos can end the utterance early, so the page
  // uses this to lay out a waveform it fills in as frames arrive.
  post('gen_start', {
    tokens: ids.length, prepared, maxFrames: max_frames_for_tokens(ids.length),
  });

  const onFrame = (pcm, index) => {
    // `pcm` is a view into wasm memory and is reused, so copy before transfer.
    const copy = new Float32Array(pcm);
    post('frame', { pcm: copy, index }, [copy.buffer]);
  };

  const stats = JSON.parse(
    await generate(voice, new Uint32Array(ids), framesAfterEos, seed, onFrame),
  );
  post('gen_done', { stats });
}

self.onmessage = async ev => {
  const { type, ...args } = ev.data;
  try {
    if (type === 'setup') await setup(args);
    else if (type === 'generate') await run(args);
  } catch (e) {
    post('error', { message: e && e.message ? e.message : String(e) });
  }
};
