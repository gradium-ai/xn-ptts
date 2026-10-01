/** Text-normalization language. `'none'` hands text to the tokenizer as written. */
export type Lang = 'en' | 'fr' | 'de' | 'es' | 'pt' | 'none';

/** Weight format. `q8` is smaller (~146 MB against ~240 MB) and faster on CPU. */
export type Quant = 'f32' | 'q8';

/** Where a checkpoint's files are. Relative URLs resolve against the page. */
export interface ModelSpec {
  /** Weights file per format: safetensors or GGUF. Only the `quant` being loaded is needed. */
  weights: Partial<Record<Quant, string | URL>>;
  /** The checkpoint's `tokenizer.json`. */
  tokenizer: string | URL;
  /** Its `config.json`, or `null` for the original Pocket TTS architecture. */
  config?: string | URL | null;
  /**
   * Voice name to voice `.safetensors` file. Fetched on first use. A model without any can
   * still be loaded, but every request then needs a voice added with `addVoice` first.
   */
  voices?: Record<string, string | URL>;
  /** Voice used when a request names none. Defaults to the first of `voices`. */
  defaultVoice?: string;
}

export interface LoadProgress {
  /** `'weights'`, `'tokenizer'`, `'config'` or `'voice:<name>'`. */
  file: string;
  loaded: number;
  /** `null` when the server did not say. */
  total: number | null;
  /** The file came from the browser cache rather than the network. */
  cached: boolean;
}

export interface LoadOptions {
  /**
   * Required. The language text is normalized as before it is tokenized: numbers, dates and
   * symbols are read out the way a speaker of that language would, which the model reads
   * noticeably better. The spoken forms differ per language, so there is no default.
   */
  lang: Lang;
  /**
   * Which word rewrites run on the normalized text: `'all'` (the default), `'none'` (or
   * `'off'`), or a comma-separated list of rule names, of which there is one today,
   * `'numbers'`. Case and surrounding space do not matter, and `'all'` / `'none'` / `'off'`
   * count only on their own, not inside a list. Ignored when `lang` is `'none'`, which does
   * not normalize at all.
   */
  rewrites?: string;
  /** Default `'q8'`. */
  quant?: Quant;
  /**
   * Required. Where the checkpoint's files are. There is no default: pass the model you
   * mean, or {@link POCKET_TTS_MODEL} for Kyutai's published Pocket TTS checkpoint.
   */
  model: ModelSpec;
  /** Voices to fetch up front. Default: just the default voice; others load on first use. */
  voices?: string[];
  /** Keep downloads in the Cache API, so the model is fetched once per browser. Default `true`. */
  cache?: boolean;
  onProgress?: (progress: LoadProgress) => void;
  /** Where `worker.js` is, for a bundler that does not pick it up from `new URL(..., import.meta.url)`. */
  workerUrl?: string | URL;
  /** Where `phonon_tts_bg.wasm` is, when it is served from somewhere other than beside the worker. */
  wasmUrl?: string | URL;
  /** The same for the threaded build, `wasm-threads/phonon_tts_bg.wasm`. */
  threadsWasmUrl?: string | URL;
  /**
   * CPU threads to generate on, counting the worker that owns the model. Default `'auto'`:
   * 3, or fewer if the device has fewer cores. Threads
   * need a cross-origin isolated page (see the README); elsewhere generation runs on one
   * thread whatever this says, and {@link PhononTTS.threadsReason} says why.
   */
  threads?: number | 'auto';
  /**
   * Where to generate. Default `'auto'`: on the GPU through WebGPU when the browser has it
   * and `quant` is `'q8'`, falling back to the CPU if WebGPU fails to start. `'webgpu'`
   * insists on the GPU and rejects if it cannot start; `'cpu'` never tries it.
   * {@link PhononTTS.deviceReason} says why the model ended up where it did.
   */
  device?: 'auto' | 'webgpu' | 'cpu';
}

export interface SpeechOptions {
  /** A name from {@link PhononTTS.voices}. Default: the model's default voice. */
  voice?: string;
  /** Sampling temperature. Default `0.3`. */
  temperature?: number;
  /** Noise seed: the same text, voice, temperature and seed give the same audio. Default `42`. */
  seed?: number;
  /** Aborting stops the generation; the stream ends with what was produced so far. */
  signal?: AbortSignal;
}

export interface SpeechStats {
  /** Sentence-aligned chunks the text was split into. */
  chunks?: number;
  tokens?: number;
  /** 80 ms frames generated. */
  frames?: number;
  samples?: number;
  /**
   * Time spent prompting the model with each chunk's text, summed. On WebGPU the prompt is
   * only queued here and runs with the chunk's first step, so this reads near zero there.
   */
  promptMs?: number;
  /** Per-frame generation time. */
  stepMs?: { avg: number; min: number; max: number };
  /** From the request starting in the worker to its first audio. */
  firstAudioMs?: number | null;
  totalMs?: number;
  /** The generation was stopped before it finished. */
  cancelled: boolean;
}

/** Audio as it is generated. Iterate it once. */
export interface SpeechStream extends AsyncIterable<Float32Array> {
  readonly sampleRate: number;
  /** Resolves with timing stats once generation ends, or rejects if it failed. */
  readonly done: Promise<SpeechStats>;
  /** Stop generating. Breaking out of a `for await` loop does the same. */
  cancel(): void;
}

export declare class PhononTTS {
  /** Download (or read from cache) a checkpoint and start it in a worker. */
  static load(options: LoadOptions): Promise<PhononTTS>;
  private constructor();

  /** Samples per second of the audio this model produces (24000 for Phonon). */
  readonly sampleRate: number;
  /** SIMD features the wasm module was built with, e.g. `{ simd128: true }`. */
  readonly features: Record<string, boolean>;
  /** Where generation runs. */
  readonly device: 'webgpu' | 'cpu';
  /** Why {@link PhononTTS.device} is what it is, e.g. `'this browser offers no WebGPU adapter'`. */
  readonly deviceReason: string;
  /** CPU threads generation runs on, counting the worker that owns the model. 1 on WebGPU. */
  readonly threads: number;
  /** Why {@link PhononTTS.threads} is what it is, e.g. `'the page is not cross-origin isolated'`. */
  readonly threadsReason: string;
  /** Voices this model can speak with: bundled ones and any added. */
  readonly voices: string[];

  /**
   * Register a voice from a voice `.safetensors` file: a precomputed embedding (`emb`) or
   * the KV-cache format of `embeddings_v2/`.
   */
  addVoice(name: string, source: string | URL | ArrayBuffer | Uint8Array | Blob): Promise<void>;
  /**
   * Speak `text`, yielding mono PCM at `sampleRate` as it is generated, 80 ms per chunk.
   * Long text is split at sentence boundaries. Requests run one at a time, in order.
   */
  stream(text: string, options?: SpeechOptions): SpeechStream;
  /** Speak `text` and return the whole waveform. */
  synth(text: string, options?: SpeechOptions): Promise<Float32Array>;
  /** Speak `text` and return a 16-bit mono WAV file. */
  synthWav(text: string, options?: SpeechOptions): Promise<Blob>;
  /** Stop the worker and free the model. Pending requests fail. */
  dispose(): void;
}

/** Kyutai's published Pocket TTS checkpoint, on Hugging Face. Pass it as `model` to use it. */
export declare const POCKET_TTS_MODEL: Readonly<ModelSpec>;

/** Delete every file this package has cached. Resolves to whether there was anything. */
export declare function clearCache(): Promise<boolean>;

/** Encode mono float PCM as a 16-bit WAV file. */
export declare function encodeWav(pcm: Float32Array, sampleRate: number): Blob;

/** Join PCM chunks into one buffer. */
export declare function concatPcm(chunks: Float32Array[]): Float32Array;
