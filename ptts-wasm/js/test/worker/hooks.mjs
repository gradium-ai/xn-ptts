// Module hooks for worker.test.mjs: worker.js gets the fake wasm module and the fake WebGPU
// engine beside this file instead of the real ones, which need a build and a GPU. Everything
// else it imports is the real thing.

const FAKES = {
  './wasm/phonon_tts.js': './wasm.mjs',
  './wasm-threads/phonon_tts.js': './wasm.mjs',
  './gpu/engine.js': './engine.mjs',
  'https://example.com/glue/phonon_tts.js': './wasm-glue.mjs',
  'https://example.com/glue/phonon_tts_threads.js': './wasm-glue.mjs',
};

export async function resolve(specifier, context, next) {
  const parent = context.parentURL ? new URL(context.parentURL).pathname : '';
  if (parent.endsWith('/js/worker.js') && specifier in FAKES) {
    return { url: new URL(FAKES[specifier], import.meta.url).href, shortCircuit: true };
  }
  return next(specifier, context);
}
