// A fake of the wasm module worker.js loads (`wasm/phonon_tts.js`), for worker.test.mjs. It
// does what the test's script in `globalThis.phononFake` says and logs every call there.

const fake = () => globalThis.phononFake;
const bytesOf = (b) => {
  // A transferred buffer is detached: reading it is the bug the tests look for.
  if (b.byteLength === 0) throw new Error('read a detached buffer');
  return new TextDecoder().decode(b);
};

export default async function init() {
  fake().log.push('wasm.init');
}

export function cpu_features() {
  return { simd128: true };
}

class Meta {
  frame_size() {
    return 1920;
  }
  sample_rate() {
    return 24000;
  }
  languages() {
    return ['en'];
  }
}

export class Model extends Meta {
  static async load(weights, tokenizer, config, quant, lang, rewrites, conditions, device) {
    for (const b of [weights, tokenizer, config]) bytesOf(b);
    fake().log.push(`Model.load ${device}`);
    if (fake().cpuFails) throw new Error(fake().cpuFails);
    return new Model(device);
  }
  constructor(device) {
    super();
    this.dev = device;
    this.voiceCount = 0;
  }
  device() {
    return this.dev;
  }
  add_voice(bytes) {
    if (bytesOf(bytes) === 'bad') throw new Error('bad voice file');
    fake().log.push(`Model.add_voice ${this.dev}`);
    return this.voiceCount++;
  }
  async flush() {}
}

export class Preparer extends Meta {
  constructor(tokenizer, config, tensors) {
    super();
    for (const b of [tokenizer, config]) bytesOf(b);
    fake().log.push('Preparer.new');
    if (fake().preparerFails) throw new Error(fake().preparerFails);
  }
  conditions() {
    return [0, 0];
  }
  voice(bytes) {
    if (bytesOf(bytes) === 'bad') throw new Error('bad voice file');
    return { kind: 'rows', frames: 1, rows: new Float32Array(2) };
  }
  free() {
    fake().log.push('Preparer.free');
  }
}
