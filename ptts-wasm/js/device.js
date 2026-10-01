// Where generation runs: the GPU through WebGPU, or the CPU. Kept free of browser globals so
// the node tests can cover it; `worker.js` passes in whether the browser offers a WebGPU
// adapter, which `navigator.gpu` alone does not say: a browser with WebGPU switched off can
// still expose it, and only fails to hand out an adapter.
//
// WebGPU is preferred when the browser has it, since the GPU takes the work off the CPU the
// page shares. It needs `q8` weights, which go to the GPU as they are: quantizing `f32`
// weights there would read every one back to the host, which a browser cannot do. When
// WebGPU is only preferred, not requested, a failure to start it falls back to the CPU.

/**
 * @param {object} env
 * @param {'auto' | 'webgpu' | 'cpu'} [env.requested] `LoadOptions.device`.
 * @param {'q8' | 'f32'} env.quant
 * @param {boolean} env.hasWebGpu Whether `navigator.gpu.requestAdapter()` returned one.
 * @returns {{ device: 'webgpu' | 'cpu', reason: string }} The device to try first, and why.
 */
export function chooseDevice({ requested = 'auto', quant, hasWebGpu }) {
  if (requested === 'cpu') return { device: 'cpu', reason: 'requested' };
  if (requested === 'webgpu') return { device: 'webgpu', reason: 'requested' };
  if (quant !== 'q8') return { device: 'cpu', reason: 'WebGPU needs q8 weights' };
  if (!hasWebGpu) return { device: 'cpu', reason: 'this browser offers no WebGPU adapter' };
  return { device: 'webgpu', reason: 'default' };
}
