// A fake of the WebGPU engine (`gpu/engine.js`), for worker.test.mjs: it starts, or fails as the
// test's script in `globalThis.phononFake` says, and logs every call there.

export class GpuEngine {
  static async create(device, { gguf }) {
    if (gguf.byteLength === 0) throw new Error('read a detached buffer');
    globalThis.phononFake.log.push('GpuEngine.create');
    if (globalThis.phononFake.engineFails) throw new Error(globalThis.phononFake.engineFails);
    return new GpuEngine();
  }
  voices = 0;
  addVoice() {
    globalThis.phononFake.log.push('GpuEngine.addVoice');
    return this.voices++;
  }
}
