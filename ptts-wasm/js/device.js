// Where generation runs: the GPU through WebGPU, or the CPU. Kept free of browser globals so
// the node tests can cover it; `worker.js` passes in what the browser's WebGPU adapter is,
// which `navigator.gpu` alone does not say: a browser with WebGPU switched off can still
// expose it, and only fails to hand out an adapter.
//
// The CPU is the default. WebGPU is opt in, since it is not faster than the CPU on every
// device: on phones the threaded CPU build can beat it. `'webgpu'` insists on it, and `'auto'`
// takes it when the browser has a hardware adapter. It needs `q8` weights in a GGUF file,
// which go to the GPU as they are: quantizing other weights there would read every one back
// to the host, which a browser cannot do. Under `'auto'` the CPU is used instead of a
// software fallback adapter, when the WebGPU engine cannot run the checkpoint (`settleDevice`),
// and when WebGPU fails to start.

/**
 * @param {object} env
 * @param {'auto' | 'webgpu' | 'cpu'} [env.requested] `LoadOptions.device`; the CPU when left out.
 * @param {'q8' | 'f32'} env.quant
 * @param {boolean} env.hasWebGpu Whether `navigator.gpu.requestAdapter()` returned one.
 * @param {boolean} [env.fallbackAdapter] Whether that adapter is a software fallback, which
 *   would run slower than the CPU path.
 * @returns {{ device: 'webgpu' | 'cpu', reason: string }} The device to try first, and why.
 */
export function chooseDevice({ requested, quant, hasWebGpu, fallbackAdapter = false }) {
  if (requested === undefined) return { device: 'cpu', reason: 'default' };
  if (requested === 'cpu') return { device: 'cpu', reason: 'requested' };
  if (requested === 'webgpu') return { device: 'webgpu', reason: 'requested' };
  if (quant !== 'q8') return { device: 'cpu', reason: 'WebGPU needs q8 weights' };
  if (!hasWebGpu) return { device: 'cpu', reason: 'this browser offers no WebGPU adapter' };
  if (fallbackAdapter) return { device: 'cpu', reason: "the browser's WebGPU adapter is a software fallback" };
  return { device: 'webgpu', reason: "'auto', and the browser has a WebGPU adapter" };
}

/**
 * The device once the WebGPU engine has read the checkpoint's config.
 * @param {object} env
 * @param {'auto' | 'webgpu' | 'cpu'} [env.requested] `LoadOptions.device`.
 * @param {{ device: 'webgpu' | 'cpu', reason: string }} env.choice `chooseDevice`'s answer.
 * @param {string | null} env.unsupported Why the engine cannot run this checkpoint, or null.
 * @returns {{ device: 'webgpu' | 'cpu', reason: string, warning?: string } | { error: string }}
 *   Under `'webgpu'` a checkpoint the engine cannot run is an error naming what it lacks;
 *   under `'auto'` the CPU runs it instead, with a warning to log.
 */
export function settleDevice({ requested, choice, unsupported }) {
  if (choice.device !== 'webgpu' || unsupported === null) return choice;
  if (requested === 'webgpu') return { error: unsupported };
  return { device: 'cpu', reason: unsupported, warning: `${unsupported}; generating on the CPU instead` };
}

/** Whether `bytes` is a GGUF file: the only weights WebGPU can load. */
export function isGguf(bytes) {
  return bytes.length >= 4 && bytes[0] === 0x47 && bytes[1] === 0x47 && bytes[2] === 0x55 && bytes[3] === 0x46;
}
