// The device policy in device.js, without a browser.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { chooseDevice, isGguf } from '../device.js';

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

test("auto: the CPU rather than a software fallback adapter, unless WebGPU is asked for", () => {
  assert.deepEqual(chooseDevice({ quant: 'q8', hasWebGpu: true, fallbackAdapter: true }), {
    device: 'cpu',
    reason: "the browser's WebGPU adapter is a software fallback",
  });
  assert.deepEqual(chooseDevice({ requested: 'webgpu', quant: 'q8', hasWebGpu: true, fallbackAdapter: true }), {
    device: 'webgpu',
    reason: 'requested',
  });
  // An explicit 'auto' behaves like the default.
  assert.deepEqual(chooseDevice({ requested: 'auto', quant: 'q8', hasWebGpu: false }), {
    device: 'cpu',
    reason: 'this browser offers no WebGPU adapter',
  });
});

test('isGguf reads the magic, not the file name', () => {
  assert.equal(isGguf(new Uint8Array([0x47, 0x47, 0x55, 0x46, 3, 0, 0, 0])), true);
  assert.equal(isGguf(new TextEncoder().encode('{"__metadata__"')), false);
  assert.equal(isGguf(new Uint8Array([0x47, 0x47])), false);
});
