// Where the default checkpoint's files live.
//
// Every URL names a pinned revision rather than `main`: the files are cached by URL, and a
// published package version has to keep loading the checkpoint it was tested with even after
// the repo moves on. Bump the revisions and the package version together.
//
// TODO: point this at the public Phonon repo once the weights are published there.

const HF = 'https://huggingface.co';
const F32_REPO = `${HF}/kyutai/pocket-tts-without-voice-cloning/resolve/8843db76457a91db32077edf8dfcd1c0e3e755fd`;
const Q8_REPO = `${HF}/lmz/pocket-tts-without-voice-cloning-q8/resolve/c2d23606a738c5afb5e24e44f9d2f5d6af1b4528`;

const VOICES = ['alba', 'marius', 'javert', 'jean', 'fantine', 'cosette', 'eponine', 'azelma'];

/** @type {import('./index.js').ModelSpec} */
export const DEFAULT_MODEL = Object.freeze({
  weights: Object.freeze({
    f32: `${F32_REPO}/tts_b6369a24.safetensors`,
    q8: `${Q8_REPO}/tts_b6369a24.gguf`,
  }),
  tokenizer: `${F32_REPO}/tokenizer.json`,
  // This checkpoint ships no config.json; the runtime's built-in one describes it.
  config: null,
  // `embeddings/` holds voice embeddings, 0.5 MB each. `embeddings_v2/` has the same voices as
  // precomputed KV caches, 6 MB each: they skip one forward pass per voice, which is not worth
  // twelve times the download.
  voices: Object.freeze(
    Object.fromEntries(VOICES.map((name) => [name, `${F32_REPO}/embeddings/${name}.safetensors`])),
  ),
  defaultVoice: 'alba',
});
