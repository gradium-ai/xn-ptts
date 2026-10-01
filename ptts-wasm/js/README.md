# phonon-tts

Text-to-speech that runs in the browser, on the user's device. No server, no API key. It streams 24 kHz speech from a [Pocket TTS](https://github.com/kyutai-labs/pocket-tts) architecture checkpoint, compiled to WebAssembly from the [Phonon](https://github.com/gradium-ai/xn-ptts) Rust runtime.

```bash
npm install phonon-tts
```

```js
import { PhononTTS } from 'phonon-tts';

const tts = await PhononTTS.load({
  lang: 'en',
  model: {
    weights: { q8: '/model/model.q8.gguf' },
    tokenizer: '/model/tokenizer.json',
    config: '/model/config.json',
    voices: { Freya: '/model/voices/Freya.safetensors' },
  },
});
const wav = await tts.synthWav('Hello from your own browser.');
new Audio(URL.createObjectURL(wav)).play();
```

`model` is required: it says where the checkpoint's files are, here a model folder served under `/model/`. See [The model](#the-model) below.

The first `load` downloads the files, whose size depends on the checkpoint. They are kept in the browser's Cache API, so later page loads start from disk.

## Streaming

`stream` yields audio as it is generated, in 80 ms chunks of mono `Float32Array` at `tts.sampleRate`. Playback can start after the first chunk.

```js
const ctx = new AudioContext({ sampleRate: tts.sampleRate });
let t = ctx.currentTime;

for await (const pcm of tts.stream('A longer piece of text. It is split at sentence boundaries.')) {
  const buffer = ctx.createBuffer(1, pcm.length, tts.sampleRate);
  buffer.getChannelData(0).set(pcm);
  const source = ctx.createBufferSource();
  source.buffer = buffer;
  source.connect(ctx.destination);
  t = Math.max(t, ctx.currentTime);
  source.start(t);
  t += buffer.duration;
}
```

Text of any length works. It is split into sentence-aligned chunks and spoken one after another.

To stop, break out of the loop, call `stream.cancel()`, or pass an `AbortSignal`:

```js
const controller = new AbortController();
const speech = tts.stream(text, { signal: controller.signal });
stopButton.onclick = () => controller.abort();
```

`speech.done` resolves with timing stats (frames, time to first audio, per-frame time) once generation ends.

## API

### `PhononTTS.load(options)`

| Option | Default | |
|---|---|---|
| `model` | **required** | where the checkpoint's files are, see [The model](#the-model) |
| `lang` | **required** | `'en'`, `'fr'`, `'de'`, `'es'`, `'pt'`, or `'none'`. Numbers, dates and symbols are read out the way a speaker of that language would say them. The spoken forms differ per language, so there is no default. `'none'` passes text through as written. |
| `rewrites` | `'all'` | which word rewrites run on the normalized text: `'all'`, `'none'`, or a comma-separated list of rule names, of which there is one today, `'numbers'`. Inert when `lang` is `'none'`. |
| `quant` | `'q8'` | `'q8'` (smaller, faster) or `'f32'` |
| `voices` | the default voice | voices to fetch during `load`. Others are fetched the first time they are used. |
| `cache` | `true` | keep downloads in the Cache API |
| `onProgress` | | `({ file, loaded, total, cached }) => void`, for a progress bar |
| `device` | `'auto'` | `'auto'`, `'webgpu'` or `'cpu'`: where to generate, see [WebGPU](#webgpu) |
| `threads` | `'auto'` | CPU threads to generate on, or `'auto'` for 3. Needs a cross-origin isolated page, see [Threads](#threads) |
| `workerUrl`, `wasmUrl`, `threadsWasmUrl` | beside `index.js` | for setups that serve the package's files from elsewhere |

### Instance

- `tts.stream(text, { voice, temperature, seed, signal })` returns a `SpeechStream`: an async iterable of `Float32Array`, plus `done` and `cancel()`.
- `tts.synth(text, options)` returns the whole waveform as a `Float32Array`.
- `tts.synthWav(text, options)` returns a WAV `Blob`.
- `tts.voices`: the names you can pass as `voice`: the keys of `model.voices`, plus any added with `addVoice`.
- `tts.addVoice(name, source)` registers a voice from a URL, `Blob` or bytes of a voice `.safetensors` file.
- `tts.sampleRate`: 24000.
- `tts.device`: `'webgpu'` or `'cpu'`, where generation runs, and `tts.deviceReason` why.
- `tts.threads`: the CPU threads generation runs on, and `tts.threadsReason` why that many.
- `tts.dispose()` stops the worker and frees the model's memory.

`temperature` defaults to `0.3` and `seed` to `42`. The same text, voice, temperature and seed always give the same audio.

Requests on one instance run one at a time, in the order they were made.

### Helpers

- `encodeWav(pcm, sampleRate)` returns a 16-bit mono WAV `Blob`.
- `concatPcm(chunks)` joins stream chunks.
- `clearCache()` deletes everything this package has cached.

## The model

`model` says where a checkpoint's files are. Relative URLs resolve against the page. There is no default: loading a checkpoint other than the one you meant still produces plausible speech, so the package never picks one for you.

```js
await PhononTTS.load({
  lang: 'fr',
  model: {
    weights: { q8: '/models/fr/model.q8.gguf', f32: '/models/fr/model.safetensors' },
    tokenizer: '/models/fr/tokenizer.json',
    config: '/models/fr/config.json',  // omit for the original Pocket TTS architecture
    voices: { anna: '/models/fr/embeddings/anna.safetensors' },
    defaultVoice: 'anna',
  },
});
```

`defaultVoice` is the voice used when a request names none; it defaults to the first of `voices`. Only the weights for the `quant` you load have to be listed.

To try the package without a checkpoint of your own, pass `POCKET_TTS_MODEL`, which points at Kyutai's published Pocket TTS checkpoint on Hugging Face (about 146 MB in `q8`, 240 MB in `f32`) and its voices `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine` and `azelma`:

```js
import { PhononTTS, POCKET_TTS_MODEL } from 'phonon-tts';

const tts = await PhononTTS.load({ lang: 'en', model: POCKET_TTS_MODEL });
```

## How it runs

The model runs in a dedicated Web Worker, on the GPU through WebGPU when the browser offers it and on the CPU otherwise. Generating never blocks the page, and the main thread only receives audio. The package is plain ES modules and needs no bundler. The worker is referenced with `new URL('./worker.js', import.meta.url)`, which Vite, webpack 5, Parcel and esbuild all recognise and bundle.

Requirements:

- A browser with WebAssembly SIMD and Relaxed SIMD, and module workers. Tested in Chrome and Firefox. A browser without Relaxed SIMD cannot load the module, and `load` rejects with an error saying so.
- A secure context (`https://` or `localhost`) for caching. Elsewhere it still works, but downloads again on every load.

How close to real time it gets depends on the device, on [WebGPU](#webgpu) and on [threads](#threads), and `q8` is noticeably faster than `f32`.

## WebGPU

By default the model runs on the GPU when the browser hands out a WebGPU adapter and the weights are `q8`, and on the CPU otherwise. `q8` weights go to the GPU as they are; `f32` weights would have to be quantized there, which means reading each one back to the host, and a browser cannot wait for that. WebGPU needs no cross-origin isolation, and it works in the same secure contexts as the cache.

On the GPU the model generates several frames per round trip to the GPU and hands them over together, so its chunks of audio are longer than the CPU's 80 ms. If WebGPU fails to start, `'auto'` falls back to the CPU and `tts.deviceReason` says why. Pass `device: 'webgpu'` to insist on the GPU, and `load` rejects instead; pass `device: 'cpu'` to never try it. The GPU computes in a different order from the CPU, so its audio is not bit-identical to the CPU's.

## Threads

On the CPU, generation runs on several threads when the page is [cross-origin isolated](https://developer.mozilla.org/docs/Web/API/Window/crossOriginIsolated), and on one otherwise. Isolation is what makes the shared memory wasm threads need available, and a page gets it by being served with these two headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

With them, the page can only load cross-origin resources that opt in, through CORS or a `Cross-Origin-Resource-Policy` header. That includes the model files, if they are served from another origin.

The package ships two wasm builds and picks one when it loads: the threaded build on an isolated page, the single-threaded one elsewhere, or if threads fail to start. Both produce the same audio. `tts.threads` says how many threads it got, and `tts.threadsReason` why. `'auto'` uses 3 threads, or fewer on a device with fewer cores: past a few threads, handing out the work costs more than it saves, and a thread that lands on an efficiency core slows the rest down. Pass `threads: 1` to stay on one thread, or a number of your own for devices you know better.

This build speaks with ready-made voices only. Cloning a voice from an audio sample needs the Mimi encoder, which is not in the browser build. Create a voice file with the `create_voice` tool from the [repository](https://github.com/gradium-ai/xn-ptts), then load it with `addVoice`.

## Licence

The package is MIT OR Apache-2.0. Model weights are not part of it and come with their own licence; for `POCKET_TTS_MODEL`, see its [model card](https://huggingface.co/kyutai/pocket-tts).
