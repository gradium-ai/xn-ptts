// The device policy in device.js, without a browser.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chooseDevice } from '../device.js';

test('auto: WebGPU when the browser has it and the weights are q8', () => {
  assert.deepEqual(chooseDevice({ quant: 'q8', hasWebGpu: true }), { device: 'webgpu', reason: 'default' });
});

test('auto: the CPU, with the reason, otherwise', () => {
  assert.deepEqual(chooseDevice({ quant: 'q8', hasWebGpu: false }), { device: 'cpu', reason: 'this browser offers no WebGPU adapter' });
  assert.deepEqual(chooseDevice({ quant: 'f32', hasWebGpu: true }), { device: 'cpu', reason: 'WebGPU needs q8 weights' });
});

test('a requested device is taken as asked', () => {
  assert.deepEqual(chooseDevice({ requested: 'cpu', quant: 'q8', hasWebGpu: true }), { device: 'cpu', reason: 'requested' });
  // The worker refuses it if the browser has no WebGPU, rather than quietly using the CPU.
  assert.deepEqual(chooseDevice({ requested: 'webgpu', quant: 'q8', hasWebGpu: false }), { device: 'webgpu', reason: 'requested' });
});
