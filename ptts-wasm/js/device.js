// Where generation runs: the GPU through WebGPU, or the CPU. Kept free of browser globals so
// the node tests can cover it; `worker.js` passes in what the browser's WebGPU adapter is,
// which `navigator.gpu` alone does not say: a browser with WebGPU switched off can still
// expose it, and only fails to hand out an adapter.
//
// WebGPU is preferred when the browser has it, since the GPU takes the work off the CPU the
// page shares. It needs `q8` weights in a GGUF file, which go to the GPU as they are:
// quantizing other weights there would read every one back to the host, which a browser
// cannot do. When WebGPU is only preferred, not requested, the CPU is used instead of a
// software fallback adapter, and when WebGPU fails to start.

/**
 * @param {object} env
 * @param {'auto' | 'webgpu' | 'cpu'} [env.requested] `LoadOptions.device`.
 * @param {'q8' | 'f32'} env.quant
 * @param {boolean} env.hasWebGpu Whether `navigator.gpu.requestAdapter()` returned one.
 * @param {boolean} [env.fallbackAdapter] Whether that adapter is a software fallback, which
 *   would run slower than the CPU path.
 * @returns {{ device: 'webgpu' | 'cpu', reason: string }} The device to try first, and why.
 */
export function chooseDevice({ requested = 'auto', quant, hasWebGpu, fallbackAdapter = false }) {
  if (requested === 'cpu') return { device: 'cpu', reason: 'requested' };
  if (requested === 'webgpu') return { device: 'webgpu', reason: 'requested' };
  if (quant !== 'q8') return { device: 'cpu', reason: 'WebGPU needs q8 weights' };
  if (!hasWebGpu) return { device: 'cpu', reason: 'this browser offers no WebGPU adapter' };
  if (fallbackAdapter) return { device: 'cpu', reason: "the browser's WebGPU adapter is a software fallback" };
  return { device: 'webgpu', reason: 'default' };
}

/** Whether `bytes` is a GGUF file: the only weights WebGPU can load. */
export function isGguf(bytes) {
  return bytes.length >= 4 && bytes[0] === 0x47 && bytes[1] === 0x47 && bytes[2] === 0x55 && bytes[3] === 0x46;
}
