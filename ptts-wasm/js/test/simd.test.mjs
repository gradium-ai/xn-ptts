// The Relaxed SIMD probe in simd.js, which picks between each build's two modules.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RELAXED_SIMD_PROBE, hasRelaxedSimd } from '../simd.js';

test('Node has Relaxed SIMD, so the probe validates', () => {
  assert.equal(hasRelaxedSimd(), true);
});

test('the probe rests on its one Relaxed SIMD instruction', () => {
  // f32x4.relaxed_madd, 0xfd 0x105, as the module's last instruction before `end`.
  const at = RELAXED_SIMD_PROBE.length - 4;
  assert.deepEqual([...RELAXED_SIMD_PROBE.subarray(at, at + 3)], [0xfd, 0x85, 0x02]);
  // With that opcode replaced by one no engine defines, the same module is invalid.
  const broken = RELAXED_SIMD_PROBE.slice();
  broken.set([0xfd, 0xff, 0x0f], at);
  assert.equal(WebAssembly.validate(broken), false);
});
