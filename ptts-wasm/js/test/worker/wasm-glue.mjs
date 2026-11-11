// A fake of a relocated glue module (`wasm/phonon_tts.js` served from elsewhere), for
// worker.test.mjs. Same shape as wasm.mjs, with its own marker and the thread-pool entry
// points, so the tests can tell which build actually loaded.

const fake = () => globalThis.phononFake;
const bytesOf = (b) => {
  if (b.byteLength === 0) throw new Error('read a detached buffer');
  return new TextDecoder().decode(b);
};

export default async function init() {
  fake().log.push('glue.wasm.init');
}

export async function initThreadPool(n) {
  fake().log.push(`glue.initThreadPool ${n}`);
}

export function start_cpu_pool(n) {
  fake().log.push(`glue.start_cpu_pool ${n}`);
  return {};
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
    fake().log.push(`GlueModel.load ${device}`);
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
    fake().log.push(`GlueModel.add_voice ${this.dev}`);
    return this.voiceCount++;
  }
  async flush() {}
}

export class Preparer extends Meta {
  constructor(tokenizer, config, tensors) {
    super();
    for (const b of [tokenizer, config]) bytesOf(b);
    fake().log.push('GluePreparer.new');
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
    fake().log.push('GluePreparer.free');
  }
}
