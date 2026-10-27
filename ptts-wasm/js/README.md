# phonon-tts

Text-to-speech that runs in the browser, on the user's device. No server, no API key. It streams speech from a Phonon checkpoint, compiled to WebAssembly from the [Rust runtime](https://github.com/gradium-ai/xn-ptts).

```bash
npm install phonon-tts
```

```js
import { PhononTTS } from 'phonon-tts';

const tts = await PhononTTS.load({ lang: 'en' });
const wav = await tts.synthWav('Hello from your own browser.');
new Audio(URL.createObjectURL(wav)).play();
```

Without `model`, `load` uses the default model, `DEFAULT_MODEL`, from the Hugging Face Hub: Phonon 31m, which speaks English. See [The model](#the-model) below for another one.

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

Long text is grouped into sentence-aligned chunks and spoken one after another. The usual target is 50 text tokens per chunk; a longer single sentence is split at up to 200 tokens, cutting even a single word over the limit between its characters.

To stop, break out of the loop, call `stream.cancel()`, or pass an `AbortSignal`:

```js
const controller = new AbortController();
const speech = tts.stream(text, { signal: controller.signal });
stopButton.onclick = () => controller.abort();
```

Cancellation discards unread stream chunks. Stop any audio already scheduled in your player separately, for example by closing its `AudioContext`. Wait for `speech.done` to finish cancellation before reusing playback resources; the next request on the same model is queued automatically.

`speech.done` resolves with timing stats (frames, time to first audio, per-frame time) once generation ends. `stats.cancelled` reports whether the worker stopped before generation finished; it can be false if cancellation arrived after the worker finished.

**Migration note:** `cancel()` now discards unread chunks even when generation has already finished. To keep all generated audio, drain the stream without cancelling it.

## API

### `PhononTTS.load(options)`

| Option | Default | |
|---|---|---|
| `model` | `DEFAULT_MODEL` | where the checkpoint's files are, see [The model](#the-model) |
| `lang` | **required** | `'en'`, `'fr'`, `'de'`, `'es'`, `'pt'`, or `'none'`. Numbers, dates and symbols are read out the way a speaker of that language would say them. The spoken forms differ per language, so there is no default. `'none'` passes text through as written. |
| `rewrites` | `'default'` | which word rewrites run on the normalized text: `'default'` (numbers, currency, dashed-digits, emails, urls, abbreviations, elongations), `'all'` (those and phones, times, dates), `'none'`, or a comma-separated list of rule names. Inert when `lang` is `'none'`. |
| `conditions` | | values for the conditioners the checkpoint's `config.json` lists, by name, e.g. `{ duration_delta: 0.3 }`. Those left out take their defaults. |
| `quant` | `'q8'` | `'q8'` (smaller, faster) or `'f32'` |
| `voices` | the default voice | voices to fetch during `load`. Others are fetched the first time they are used. |
| `cache` | `true` | keep downloads in the Cache API |
| `onProgress` | | `({ file, loaded, total, cached }) => void`, for a progress bar |
| `device` | `'cpu'` | `'cpu'`, `'webgpu'` or `'auto'`: where to generate, see [WebGPU](#webgpu) |
| `threads` | `'auto'` | CPU threads to generate on, or `'auto'` for 3. Needs a cross-origin isolated page, see [Threads](#threads) |
| `workerUrl`, `wasmUrl`, `threadsWasmUrl`, `simdWasmUrl`, `threadsSimdWasmUrl` | beside `index.js` | for setups that serve the package's files from elsewhere |

### Instance

- `tts.stream(text, { voice, temperature, seed, signal })` returns a `SpeechStream`: an async iterable of `Float32Array`, plus `done` and `cancel()`.
- `tts.synth(text, options)` returns the whole waveform as a `Float32Array`.
- `tts.synthWav(text, options)` returns a WAV `Blob`.
- `tts.voices`: the names you can pass as `voice`: the keys of `model.voices`, plus any added with `addVoice`.
- `tts.addVoice(name, source)` registers a voice from a URL, `Blob` or bytes of a voice `.safetensors` file.
- `tts.languages`: the languages the checkpoint's config says it was trained for.
- `tts.sampleRate`: 24000.
- `tts.device`: `'webgpu'` or `'cpu'`, where generation runs, and `tts.deviceReason` why.
- `tts.threads`: the CPU threads generation runs on, and `tts.threadsReason` why that many.
- `tts.dispose()` stops the worker and frees the model's memory.

`temperature` defaults to `0.3` and `seed` to `4242424242424242`, the same defaults as the Rust and Python packages. The same text, voice, temperature and seed always give the same audio. A seed above `Number.MAX_SAFE_INTEGER` has to be a `bigint`.

Requests on one instance run one at a time, in the order they were made.

### Helpers

- `encodeWav(pcm, sampleRate)` returns a 16-bit mono WAV `Blob`.
- `concatPcm(chunks)` joins stream chunks.
- `clearCache()` deletes everything this package has cached.

## The model

`model` says where a checkpoint's files are. Relative URLs resolve against the page. Left out, it is `DEFAULT_MODEL`, which the package exports: Phonon 31m (English), a public Hugging Face repo pinned to a commit.

```js
await PhononTTS.load({
  lang: 'fr',
  model: {
    weights: { q8: '/models/fr/model.q8.gguf', f32: '/models/fr/model.safetensors' },
    tokenizer: '/models/fr/tokenizer.json',
    config: '/models/fr/config.json',  // required for every checkpoint
    voices: { anna: '/models/fr/embeddings/anna.safetensors' },
    defaultVoice: 'anna',
  },
});
```

`defaultVoice` is the voice used when a request names none; it defaults to the first of `voices`. Only the weights for the `quant` you load have to be listed.

A public Hugging Face repo can be named instead, as `hf://owner/name` or `hf://owner/name@revision` (a branch, tag or commit; `main` when left out):

```js
await PhononTTS.load({ lang: 'en', model: 'hf://gradium/phonon-31m-en@COMMIT_SHA' });
```

One request to the Hub finds the commit and the repo's files, and every file URL is pinned to that commit, so a cached file is never served after the repo changes. The files are picked as the native frontends pick them: `config.json`, `tokenizer.json`, `model.q8.gguf` or `model.safetensors`, and voices under `voices/` or `embeddings/` plus `default-voice.safetensors` as `default`. The Hub's answer is cached too, so a model loaded once also loads offline, or while the Hub fails; a branch then means the commit it pointed at when last looked up. Private and gated repos cannot be loaded this way: the browser has no token to send. That includes the Phonon 40m models. To use one, accept its terms on its Hugging Face page, download its files, and serve them with your app through a `ModelSpec` as above, under the model's license.

Every model needs its own `config.json` and `tokenizer.json`. Supply their URLs in `ModelSpec`, or a repo that has them; the package has no model or config fallback. Pocket TTS checkpoints work through this same interface when they supply those files and compatible weights.

## How it runs

The model runs in a dedicated Web Worker, on the CPU unless you ask for the GPU (see [WebGPU](#webgpu)). Generating never blocks the page, and the main thread only receives audio. The package is plain ES modules and needs no bundler. The worker is referenced with `new URL('./worker.js', import.meta.url)`, which Vite, webpack 5, Parcel and esbuild all recognise and bundle.

Requirements:

- A browser with WebAssembly SIMD and module workers. Browsers with Relaxed SIMD get a faster build; others, such as Safari and every browser on iOS, load one without it. A browser without SIMD cannot load the module, and `load` rejects with an error saying so.
- A secure context (`https://` or `localhost`) for caching. Elsewhere it still works, but downloads again on every load.

How close to real time it gets depends on the device, on [WebGPU](#webgpu) and on [threads](#threads), and `q8` is noticeably faster than `f32`.

## WebGPU

The model runs on the CPU unless asked otherwise. WebGPU is opt in, because it is not faster than the CPU on every device: on phones the threaded CPU build can beat it. Pass `device: 'webgpu'` to run on the GPU, and `load` rejects if it cannot start. Pass `device: 'auto'` to use the GPU when the browser hands out a WebGPU adapter and the weights are `q8` in a GGUF file, and the CPU otherwise. Under `'auto'` a software fallback adapter counts as none, since it would run slower than the CPU. `q8` weights go to the GPU as they are; other weights would have to be quantized there, which means reading each one back to the host, and a browser cannot wait for that. WebGPU needs no cross-origin isolation, and it works in the same secure contexts as the cache.

On the GPU the model generates several frames per round trip to the GPU and hands them over together, so its chunks of audio are longer than the CPU's 80 ms. If WebGPU fails to start, `'auto'` falls back to the CPU and `tts.deviceReason` says why. That CPU run stays on one thread: it uses the build WebGPU was loaded in, rather than copying the weights into a second one. The GPU computes in a different order from the CPU, so its audio is not bit-identical to the CPU's.

## Threads

On the CPU, generation runs on several threads when the page is [cross-origin isolated](https://developer.mozilla.org/docs/Web/API/Window/crossOriginIsolated), and on one otherwise. Isolation is what makes the shared memory wasm threads need available, and a page gets it by being served with these two headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

With them, the page can only load cross-origin resources that opt in, through CORS or a `Cross-Origin-Resource-Policy` header. That includes the model files, if they are served from another origin.

The package ships two wasm builds, each with and without Relaxed SIMD, and picks one when it loads: the threaded build on an isolated page, the single-threaded one elsewhere, or if threads fail to start. Both produce the same audio. `tts.threads` says how many threads it got, and `tts.threadsReason` why. `'auto'` uses 3 threads, or fewer on a device with fewer cores: past a few threads, handing out the work costs more than it saves, and a thread that lands on an efficiency core slows the rest down. Pass `threads: 1` to stay on one thread, or a number of your own for devices you know better.

This build speaks with ready-made voices only: cloning from an audio sample is not in the browser build. Voices from the catalog [`gradium/phonon-voices`](https://huggingface.co/gradium/phonon-voices) load with `addVoice` from the URL of their file, for example `await tts.addVoice('Zoey', 'https://huggingface.co/gradium/phonon-voices/resolve/main/voices/NbpkqMVS3CJeq2j8.safetensors')`. The catalog's `voices.jsonl` lists each voice's name, language and file. Voice files from the Gradium Studio scripts in the [repository](https://github.com/gradium-ai/xn-ptts#voices) load the same way.

## Licence

The package is MIT OR Apache-2.0. The Rust crates compiled into its WebAssembly modules keep their own licences, and their notices are in `THIRD_PARTY_LICENSES.md`. Model weights are not part of it and come with their own licenses: Phonon 31m is under Apache 2.0, and Phonon 40m and Phonon 40m multilingual are under CC BY-NC 4.0. See each model's page on Hugging Face.
