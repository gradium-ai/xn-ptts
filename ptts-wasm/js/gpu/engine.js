// The WebGPU engine: token ids and voice rows in, 24 kHz PCM out, with the flow LM and the
// Mimi decoder both on the device and nothing read back but the audio.
//
// What runs where
//
//   Rust (the wasm `Preparer`): text normalization, `plan::chunks`, the tokenizer, voice files,
//   the conditioner sum. This file: the weights (read from the GGUF by weights.js, never copied
//   into wasm), the device buffers, and the loop below. The kernels are in kernels.js.
//
// Dispatch lists (lm.js, codec.js, dispatch.js)
//
//   Every piece of work is a fixed list of dispatches whose pipelines, bind groups and uniforms
//   are made once at load (pipelines with createComputePipelineAsync, all compiled before the
//   load resolves). Recording is setPipeline / setBindGroup / dispatchWorkgroups only.
//
//   step       one LM frame: per layer, a LayerNorm-fused QKV GEMV, split-KV attention that
//              also writes the new key and value (f16 cache), the out-projection GEMV merging
//              the attention splits, LayerNorm-fused FC1 and FC2; then the flow head (the
//              conditioning GEMV, one GEMV for all the adaLN modulations, the noise drawn on the
//              GPU into the input projection, two GEMVs per residual block, the final layer)
//              and a finish kernel (EOS logit, state update, next input row). 6 x 5 + 17 = 47
//              dispatches for the 6-layer model, 62 for the 9-layer one.
//   prompt     a voice's or a chunk's rows through every layer at once: LayerNorm, GEMMs (q8
//              weights), RoPE with the f16 cache write, causal attention over the cache, 8 per
//              layer, then one that advances the position: 49 (6 layers) or 73 (9), however
//              many rows the prompt has.
//   codecC     one chunk of C = 1, 2, 4 or 8 frames through Mimi: 29 dispatches (codec.js).
//
// State buffer (`st`, 16 u32, slots in kernels.js `ST`)
//
//   pos        KV position the next step writes        host at chunk start, then the GPU
//   frame      frame the next step makes                GPU
//   end        frames to keep, 0 until end of speech    GPU (finish: eos frame + 1 + after)
//   stop       set at `end` or at `limit`              GPU; every LM kernel returns when set
//   seed       noise seed (u32), per chunk              host
//   scale      sqrt(temperature), f32 bits              host
//   after      frames kept after the end-of-speech frame (`plan::Chunk::frames_after_eos`)
//   limit      the chunk's frame budget                 host
//   cpos       first frame of the codec chunk           host, with writeBuffer before a submit
//   rows       rows of the prompt being run             host
//
//   The end-of-speech rule is `EosPolicy`'s: the frame whose logit passes `eos_threshold` is
//   kept, then `frames_after_eos` more. Codec kernels return when `cpos >= end`.
//
// Submit pipeline (speakChunk and `ChunkRun` below)
//
//   A chunk is a series of submits, up to IN_FLIGHT = 3 of them on the queue at once. Each one
//   records the LM steps of exactly the frames it decodes, so no decode reads a latent before
//   its step has run, then that codec chunk, then copies the state (64 bytes) and the chunk's
//   PCM into a readback buffer of its own, which is then mapped. The first submit also copies
//   the voice's keys and values into the cache, clears the codec's context and runs the text
//   rows, which the host wrote along with the first input row and the state. `st.cpos` is
//   written just before each submit: queue order puts it after the earlier submits' decodes.
//
//   Codec chunks grow 1, 2, 4, then 8 frames, so the first audio comes after one frame. No
//   decode reaches past the chunk's frame limit, so no more steps are recorded than it allows.
//   The host takes the readbacks in submit order and hands each one's PCM to `onAudio`, as a
//   copy, as soon as it arrives, while the GPU works through the two submits behind it.
//
//   The GPU decides the end. Once a readback shows it (`end`, or `stop` at the limit), the host
//   trims the PCM to it, issues nothing past it, and sizes what is still missing with the
//   smallest chunk that covers it. The submits queued before the host knew still complete;
//   their steps and decodes past the end return at once. Cancelling stops issuing, waits for
//   the submits in flight and drops their audio. After an error every readback buffer is
//   unmapped again, its pending map aborted, so the next call starts clean.
//
// Reductions and subgroups
//
//   With subgroups exactly 32 wide (support.js `chooseReductions`) the kernels reduce with
//   subgroupAdd, subgroupMax and subgroupShuffleXor over 32-lane groups, taking
//   `local_invocation_index >> 5` as the subgroup. Elsewhere (no subgroups, or another width:
//   Intel, AMD wave64, SwiftShader) they use workgroup shared memory. Nothing else needs an
//   optional feature: int8 weights are unpacked with shifts and f16 with unpack2x16float, and
//   every limit is the default.
//
// What it runs: support.js. Precision: q8_0 LM linears as the GGUF has them (f32 activations),
// the flow net and Mimi in f16 weights with f32 math, an f16 LM KV cache, an f32 Mimi KV ring.

import { parseGguf } from './gguf.js';
import { buildWeights, toHalf } from './weights.js';
import { GpuLM, S_MAX, F_MAX, ROWS_MAX } from './lm.js';
import { GpuCodec, CHUNKS, FRAME } from './codec.js';
import { ST, ST_WORDS } from './kernels.js';
import { DispatchBuilder } from './dispatch.js';
import { unsupportedConfig, unsupportedWeights } from './support.js';
const D = 512;
/** Submits a chunk keeps on the queue at once. */
const IN_FLIGHT = 3;
/** A readback holds the state words, then the PCM of up to the largest codec chunk. */
const STATE_BYTES = ST_WORDS * 4;
const READBACK_BYTES = STATE_BYTES + CHUNKS[CHUNKS.length - 1] * FRAME * 4;

export class GpuEngine {
  /**
   * @param {GPUDevice} device
   * @param {{ gguf: Uint8Array, config: object, conditions: Float32Array, subgroups: boolean }} o
   *   `conditions` is the Rust `Preparer`'s conditioner sum.
   */
  static async create(device, { gguf, config, conditions, subgroups }) {
    const why = unsupportedConfig(config);
    if (why) throw new Error(why);
    const g = parseGguf(gguf);
    const bad = unsupportedWeights(g, config);
    if (bad) throw new Error(bad);
    const w = buildWeights(g, config);
    const words = w.blob.finish();
    // Every kernel binds the whole blob, so it must fit one storage binding.
    const limit = device.limits.maxStorageBufferBindingSize;
    if (words.byteLength > limit) {
      throw new Error(`the weights take ${words.byteLength} bytes on the GPU, more than the ${limit} one storage binding allows here`);
    }
    const W = device.createBuffer({ label: 'weights', size: words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(W, 0, words);
    device.queue.writeBuffer(W, w.top.cond * 4, conditions);
    const lm = new GpuLM(device, w, W, { subgroups, eosThreshold: config.eos_threshold ?? 4, layers: config.flow_lm.num_layers });
    const codec = new GpuCodec(lm, w.codec, { subgroups });
    const b = new DispatchBuilder(device);
    lm.addLists(b);
    codec.addLists(b);
    const lists = await b.build();
    lm.setLists(lists);
    codec.setLists(lists);
    return new GpuEngine(device, w, lm, codec, conditions);
  }

  constructor(device, w, lm, codec, conditions) {
    this.device = device;
    this.w = w;
    this.lm = lm;
    this.codec = codec;
    this.x0 = lm.inputRow(w.host.bos, conditions);
    this.voices = [];
    // One per submit in flight. Every chunk unmaps them all before it returns.
    this.readbacks = Array.from({ length: IN_FLIGHT }, () =>
      device.createBuffer({ label: 'readback', size: READBACK_BYTES, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }));
  }

  /** Voices and chunks take the device's buffers in turn: one runs at a time, in call order. */
  #turn = Promise.resolve();
  #exclusive(fn) {
    const run = this.#turn.then(fn);
    this.#turn = run.catch(() => {});
    return run;
  }

  /** Dispatches in one LM step, one prompt and one codec chunk, for the stats. */
  get dispatches() {
    return { step: this.lm.stepList.length, prompt: this.lm.promptList.length, codecChunk: this.codec.dispatchesPerChunk(8) };
  }

  /**
   * Registers a voice from what `Preparer.voice` returns: prompt rows, run through the layers
   * once here, or a stored KV cache, uploaded as it is. Resolves to its index once the GPU has
   * run it, so a failure shows here rather than in the first generation.
   */
  addVoice(prompt) {
    return this.#exclusive(() => this.#addVoice(prompt));
  }

  async #addVoice(prompt) {
    const { lm, device } = this;
    const frames = prompt.frames;
    if (frames > ROWS_MAX) throw new Error(`the voice prompt has ${frames} frames, more than the ${ROWS_MAX} the WebGPU engine takes`);
    if (prompt.kind === 'kv') {
      if (prompt.k.length !== lm.layers || prompt.v.length !== lm.layers) {
        throw new Error(`the voice's KV cache has ${prompt.k.length} key and ${prompt.v.length} value layers, the model ${lm.layers}`);
      }
      // Rows of D values each: another head layout would read past the end as NaN.
      for (const t of [...prompt.k, ...prompt.v]) {
        if (t.length !== frames * D) throw new Error(`the voice's KV cache has ${t.length} values in a layer, not ${frames} frames of ${D}`);
      }
    }
    const bytesV = frames * D * 2;
    const kv = device.createBuffer({ label: 'voice', size: Math.max(16, 2 * lm.layers * bytesV), usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (prompt.kind === 'kv') {
      for (let l = 0; l < lm.layers; l++) {
        device.queue.writeBuffer(kv, 2 * l * bytesV, halves(prompt.k[l], frames * D));
        device.queue.writeBuffer(kv, (2 * l + 1) * bytesV, halves(prompt.v[l], frames * D));
      }
    } else {
      device.queue.writeBuffer(lm.ROWS, 0, prompt.rows);
      device.queue.writeBuffer(lm.st, 0, lm.stateWords({ pos: 0, rows: frames }));
      const enc = device.createCommandEncoder();
      const pass = enc.beginComputePass();
      lm.encodePrompt(pass, frames);
      pass.end();
      const layer = GpuLM.layerBytes();
      for (let l = 0; l < lm.layers; l++) {
        enc.copyBufferToBuffer(lm.KC, l * layer, kv, 2 * l * bytesV, bytesV);
        enc.copyBufferToBuffer(lm.VC, l * layer, kv, (2 * l + 1) * bytesV, bytesV);
      }
      device.queue.submit([enc.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    this.voices.push({ kv, frames });
    return this.voices.length - 1;
  }

  /**
   * Speaks one chunk of `Preparer.plan`: `{ tokens, framesAfterEos, frameBudget }`. `onAudio`
   * gets each piece of PCM as it is read back. Resolves to `{ frames, cancelled }`, `frames`
   * being the frames delivered, once the chunk ends, at the end of speech or the frame budget,
   * or once `cancelled()` turns true.
   */
  speakChunk(chunk, { voice, seed, temperature, onAudio, cancelled = () => false }) {
    return this.#exclusive(() => this.#speakChunk(chunk, { voice, seed, temperature, onAudio, cancelled }));
  }

  async #speakChunk(chunk, { voice, seed, temperature, onAudio, cancelled }) {
    const { device, lm } = this;
    const v = this.voices[voice];
    if (!v) throw new Error(`unknown voice index ${voice}`);
    const n = chunk.tokens.length;
    if (n > ROWS_MAX) throw new Error(`a chunk of ${n} tokens is more than the ${ROWS_MAX} the WebGPU engine prompts at once`);
    const maxFrames = Math.min(chunk.frameBudget, F_MAX, S_MAX - v.frames - n - 1);
    const table = this.w.host.embed;
    const rows = new Float32Array(Math.max(1, n) * D);
    chunk.tokens.forEach((id, r) => rows.set(table.subarray(id * D, id * D + D), r * D));
    device.queue.writeBuffer(lm.ROWS, 0, rows);
    device.queue.writeBuffer(lm.X, 0, this.x0);
    device.queue.writeBuffer(lm.st, 0, lm.stateWords({ pos: v.frames, rows: n, seed, temperature, after: chunk.framesAfterEos, limit: maxFrames }));
    return new ChunkRun(this, { voice: v, tokens: n, maxFrames, onAudio, cancelled }).run();
  }
}

/**
 * One chunk through the submit pipeline (see "Submit pipeline" above): issues the submits,
 * takes their readbacks in order, and hands the audio on until the end the GPU found.
 */
class ChunkRun {
  /** Frames issued so far. Each submit records the LM steps of exactly the frames it decodes,
   *  so this is both the steps recorded and the first frame of the next decode. */
  issued = 0;
  /** Decodes issued: the next one's place in the 1, 2, 4, 8 progression. */
  decodes = 0;
  /** Frames to keep, once a readback has shown the end; null until then. */
  keep = null;
  /** Frames handed to `onAudio`. */
  delivered = 0;
  /** Submits on the queue, oldest first: `{ buffer, mapped, first, frames, bytes }`. */
  flight = [];

  constructor(engine, { voice, tokens, maxFrames, onAudio, cancelled }) {
    this.engine = engine;
    this.voice = voice;
    this.tokens = tokens;
    this.maxFrames = maxFrames;
    this.onAudio = onAudio;
    this.cancelled = cancelled;
    /** The readback buffers not on the queue. */
    this.free = [...engine.readbacks];
  }

  /** Resolves to `{ frames, cancelled }` once every submit it issued has been read back. */
  async run() {
    // Once cancelled, nothing more is issued or delivered, but the submits in flight are still
    // waited for, so that their buffers are unmapped when this returns.
    let stopped = this.cancelled();
    try {
      for (;;) {
        // Frames to decode in all: up to the end once it is known, up to the limit until then.
        const target = this.keep ?? this.maxFrames;
        while (!stopped && this.flight.length < IN_FLIGHT && this.issued < target) this.#issue();
        const oldest = this.flight[0];
        if (!oldest) break;
        await oldest.mapped;
        if (!stopped && this.cancelled()) stopped = true;
        const pcm = this.#take(oldest, !stopped);
        if (pcm) {
          this.delivered += pcm.length / FRAME;
          this.onAudio(pcm);
        }
      }
    } finally {
      // Empty unless something threw. Unmapping aborts a map still pending, so every buffer is
      // unmapped again for the next call; the copies still queued into them land before its own.
      for (const r of this.flight) if (r.buffer.mapState !== 'unmapped') r.buffer.unmap();
    }
    return { frames: this.delivered, cancelled: stopped };
  }

  /** Records the next decode, the steps it needs and its readback, submits it, and maps it. */
  #issue() {
    const { device, lm, codec } = this.engine;
    const first = this.issued;
    const frames = this.#nextSize();
    const bytes = STATE_BYTES + frames * FRAME * 4;
    const buffer = this.free.pop();
    const enc = device.createCommandEncoder();
    if (this.decodes === 0) this.#start(enc);
    const pass = enc.beginComputePass();
    if (this.decodes === 0 && this.tokens > 0) lm.encodePrompt(pass, this.tokens);
    lm.encodeSteps(pass, frames);
    codec.encode(pass, frames);
    pass.end();
    enc.copyBufferToBuffer(lm.st, 0, buffer, 0, STATE_BYTES);
    enc.copyBufferToBuffer(codec.audio, 0, buffer, STATE_BYTES, frames * FRAME * 4);
    // The decode's first frame: on the queue behind the earlier submits, which have read theirs.
    device.queue.writeBuffer(lm.st, ST.cpos * 4, new Uint32Array([first]));
    device.queue.submit([enc.finish()]);
    const mapped = buffer.mapAsync(GPUMapMode.READ, 0, bytes);
    // Awaited in turn by `run`; this only keeps one left behind by an error from being reported.
    mapped.catch(() => {});
    this.flight.push({ buffer, mapped, first, frames, bytes });
    this.issued += frames;
    this.decodes++;
  }

  /** The chunk's first GPU work: the voice's keys and values into the cache, layer by layer,
   *  and the codec's streaming context cleared. */
  #start(enc) {
    const { lm, codec } = this.engine;
    const { kv, frames } = this.voice;
    const bytes = frames * D * 2;
    const layer = GpuLM.layerBytes();
    if (bytes > 0) {
      for (let l = 0; l < lm.layers; l++) {
        enc.copyBufferToBuffer(kv, 2 * l * bytes, lm.KC, l * layer, bytes);
        enc.copyBufferToBuffer(kv, (2 * l + 1) * bytes, lm.VC, l * layer, bytes);
      }
    }
    codec.reset(enc);
  }

  /**
   * Frames in the next decode: 1, 2, 4, then 8, cut to what the frame limit leaves, so that no
   * more steps are recorded than it allows. Once the end is known, the smallest size that
   * covers the frames still missing, when one fits under the limit.
   */
  #nextSize() {
    const fits = CHUNKS.filter((c) => this.issued + c <= this.maxFrames);
    if (this.keep !== null) {
      const cover = fits.find((c) => this.issued + c >= this.keep);
      if (cover !== undefined) return cover;
    }
    const grow = CHUNKS[Math.min(this.decodes, CHUNKS.length - 1)];
    return fits.filter((c) => c <= grow).at(-1);
  }

  /**
   * Takes the oldest submit, now mapped, off the queue and unmaps it. With `read`, it first
   * learns the end from its state and returns a copy of its PCM up to the end, or null when
   * none of it is kept.
   */
  #take(r, read) {
    let pcm = null;
    try {
      if (read) {
        const range = r.buffer.getMappedRange(0, r.bytes);
        this.#learnEnd(new Uint32Array(range, 0, ST_WORDS));
        const frames = Math.min(r.frames, (this.keep ?? Infinity) - r.first);
        if (frames > 0) pcm = new Float32Array(range, STATE_BYTES, frames * FRAME).slice();
      }
    } finally {
      r.buffer.unmap();
      this.flight.shift();
      this.free.push(r.buffer);
    }
    return pcm;
  }

  /** The frames to keep, from a readback's state: up to the end of speech once the GPU has
   *  seen it, or what was made when it stopped at the limit. Never more than `maxFrames`. */
  #learnEnd(st) {
    if (this.keep !== null) return;
    if (st[ST.end] !== 0) this.keep = Math.min(st[ST.end], this.maxFrames);
    else if (st[ST.stop] !== 0) this.keep = Math.min(st[ST.frame], this.maxFrames);
  }
}

/** f32 values as packed f16 pairs. */
function halves(f, n) {
  const out = new Uint16Array(n + (n & 1));
  for (let i = 0; i < n; i++) out[i] = toHalf(f[i]);
  return new Uint32Array(out.buffer);
}
