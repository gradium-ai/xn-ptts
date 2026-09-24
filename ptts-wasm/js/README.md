# phonon-tts

Text-to-speech that runs in the browser, on the user's device. No server, no API key. It streams 24 kHz speech from the [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) model, compiled to WebAssembly from the [Phonon](https://github.com/gradium-ai/xn-ptts) Rust runtime.

```bash
npm install phonon-tts
```

```js
import { PhononTTS } from 'phonon-tts';

const tts = await PhononTTS.load({ lang: 'en' });
const wav = await tts.synthWav('Hello from your own browser.');
new Audio(URL.createObjectURL(wav)).play();
```

The first `load` downloads the model: about 146 MB for the default `q8` weights, or 240 MB for `f32`. The files are kept in the browser's Cache API, so later page loads start from disk.

## Streaming

`stream` yields audio as it is generated, in 80 ms chunks of mono `Float32Array` at `tts.sampleRate`. Playback can start after the first chunk.

```js
const ctx = new AudioContext({ sampleRate: tts.sampleRate });
let t = ctx.currentTime;

for await (const pcm of tts.stream('A longer piece of text. It is split at sentence boundaries.', { voice: 'marius' })) {
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
| `lang` | **required** | `'en'`, `'fr'`, `'de'`, `'es'`, `'pt'`, or `'none'`. Numbers, dates and symbols are read out the way a speaker of that language would say them. The spoken forms differ per language, so there is no default. `'none'` passes text through as written. |
| `quant` | `'q8'` | `'q8'` (smaller, faster) or `'f32'` |
| `voices` | the default voice | voices to fetch during `load`. Others are fetched the first time they are used. |
| `cache` | `true` | keep downloads in the Cache API |
| `onProgress` | | `({ file, loaded, total, cached }) => void`, for a progress bar |
| `model` | `DEFAULT_MODEL` | another checkpoint, see below |
| `workerUrl`, `wasmUrl` | beside `index.js` | for setups that serve the package's files from elsewhere |

### Instance

- `tts.stream(text, { voice, temperature, seed, signal })` returns a `SpeechStream`: an async iterable of `Float32Array`, plus `done` and `cancel()`.
- `tts.synth(text, options)` returns the whole waveform as a `Float32Array`.
- `tts.synthWav(text, options)` returns a WAV `Blob`.
- `tts.voices`: the names you can pass as `voice`. Default voices: `alba`, `marius`, `javert`, `jean`, `fantine`, `cosette`, `eponine`, `azelma`.
- `tts.addVoice(name, source)` registers a voice from a URL, `Blob` or bytes of a voice `.safetensors` file.
- `tts.sampleRate`: 24000.
- `tts.dispose()` stops the worker and frees the model's memory.

`temperature` defaults to `0.3` and `seed` to `42`. The same text, voice, temperature and seed always give the same audio.

Requests on one instance run one at a time, in the order they were made.

### Helpers

- `encodeWav(pcm, sampleRate)` returns a 16-bit mono WAV `Blob`.
- `concatPcm(chunks)` joins stream chunks.
- `clearCache()` deletes everything this package has cached.

## Other checkpoints

`model` says where a checkpoint's files are. Relative URLs resolve against the page.

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

## How it runs

The model runs in a dedicated Web Worker. Generating never blocks the page, and the main thread only receives audio. The package is plain ES modules and needs no bundler. The worker is referenced with `new URL('./worker.js', import.meta.url)`, which Vite, webpack 5, Parcel and esbuild all recognise and bundle.

Requirements:

- A browser with WebAssembly SIMD and Relaxed SIMD, and module workers. Tested in Chrome and Firefox. A browser without Relaxed SIMD cannot load the module, and `load` rejects with an error saying so.
- A secure context (`https://` or `localhost`) for caching. Elsewhere it still works, but downloads again on every load.

Generation runs on one CPU thread. How close to real time it gets depends on the device, and `q8` is noticeably faster than `f32`.

This build speaks with ready-made voices only. Cloning a voice from an audio sample needs the Mimi encoder, which is not in the browser build. Create a voice file with the `create_voice` tool from the [repository](https://github.com/gradium-ai/xn-ptts), then load it with `addVoice`.

## Licence

The package is MIT OR Apache-2.0. The model weights have their own licence. See the [model card](https://huggingface.co/kyutai/pocket-tts).
