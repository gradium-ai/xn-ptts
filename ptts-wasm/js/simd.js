// Whether the browser runs WebAssembly Relaxed SIMD, which picks between the package's two
// builds of each module. Safari does not, so neither does any browser on iOS.

/**
 * A module using one Relaxed SIMD instruction. `WebAssembly.validate` rejects it where the
 * instruction is missing, which is the only reliable way to ask: there is no feature list.
 */
export const RELAXED_SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 61, 1, 59, 0, 253, 12,
  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 253, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 253, 133, 2, 11,
]);

/** Whether this browser can compile the Relaxed SIMD builds. */
export function hasRelaxedSimd() {
  try {
    return WebAssembly.validate(RELAXED_SIMD_PROBE);
  } catch {
    return false;
  }
}
