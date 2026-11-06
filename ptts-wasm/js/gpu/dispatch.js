// Fixed dispatch lists: pipelines, bind groups and uniforms are made once, so recording a list
// is only setPipeline / setBindGroup / dispatchWorkgroups. Every dispatch has its own slot of
// one uniform buffer (`P` in kernels.js).

import { P_FIELDS, P_BYTES } from './kernels.js';

const SLOT = 256; // minUniformBufferOffsetAlignment's default

export class DispatchBuilder {
  constructor(device) {
    this.device = device;
    this.layouts = new Map();
    this.kinds = new Map();
    this.pipes = new Map();
    this.lists = new Map();
  }

  /**
   * A bind group layout: storage bindings of the given kinds ('ro' or 'rw'), then the uniform.
   * lm.js and codec.js share layouts by name, so asking for a name again with other kinds is
   * an error rather than a silent reuse of the first.
   */
  layout(name, kinds) {
    const known = this.kinds.get(name);
    if (known !== undefined && known !== kinds.join()) {
      throw new Error(`bind group layout '${name}' asked for as [${kinds}] after [${known}]`);
    }
    this.kinds.set(name, kinds.join());
    if (!this.layouts.has(name)) {
      this.layouts.set(name, this.device.createBindGroupLayout({
        label: name,
        entries: [
          ...kinds.map((k, binding) => ({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: k === 'ro' ? 'read-only-storage' : 'storage' } })),
          { binding: kinds.length, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        ],
      }));
    }
    return name;
  }

  /** A pipeline for `code` with layout `layout`, compiled in the background; `key` dedupes. */
  pipe(key, layout, code) {
    if (!this.pipes.has(key)) {
      const { device } = this;
      this.pipes.set(key, {
        layout,
        promise: device.createComputePipelineAsync({
          label: key,
          layout: device.createPipelineLayout({ bindGroupLayouts: [this.layouts.get(layout)] }),
          compute: { module: device.createShaderModule({ code, label: key }), entryPoint: 'main' },
        }),
      });
    }
    return key;
  }

  /** Appends a dispatch to list `list`. `groups` is [x, y?, z?], or a function of the value
   *  `encode` is given (the rows of a prompt). `bind` holds the storage buffers in order. */
  add(list, key, groups, params, bind) {
    for (const f of Object.keys(params)) if (!P_FIELDS.includes(f)) throw new Error(`${key}: no uniform field ${f}`);
    if (!this.lists.has(list)) this.lists.set(list, []);
    this.lists.get(list).push({ key, groups, params, bind });
  }

  /** Waits for every pipeline and makes the bind groups. Returns { name: dispatches }. */
  async build() {
    const { device } = this;
    const pipes = new Map();
    const done = await Promise.all([...this.pipes].map(async ([k, p]) => [k, { pipeline: await p.promise, layout: p.layout }]));
    for (const [k, v] of done) pipes.set(k, v);
    const all = [...this.lists.values()].flat();
    const ub = device.createBuffer({ label: 'params', size: Math.max(SLOT, all.length * SLOT), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const data = new Uint32Array(all.length * (SLOT / 4));
    all.forEach((e, i) => P_FIELDS.forEach((f, j) => { data[i * (SLOT / 4) + j] = e.params[f] ?? 0; }));
    device.queue.writeBuffer(ub, 0, data);
    const out = {};
    let i = 0;
    for (const [name, entries] of this.lists) {
      out[name] = entries.map((e) => {
        const { pipeline, layout } = pipes.get(e.key);
        const bg = device.createBindGroup({
          label: e.key,
          layout: this.layouts.get(layout),
          entries: [
            ...e.bind.map((buffer, binding) => ({ binding, resource: { buffer } })),
            { binding: e.bind.length, resource: { buffer: ub, offset: i * SLOT, size: P_BYTES } },
          ],
        });
        i++;
        const fixed = typeof e.groups === 'function' ? null : e.groups;
        return { pipeline, bg, fn: fixed ? null : e.groups, x: fixed?.[0], y: fixed?.[1] ?? 1, z: fixed?.[2] ?? 1 };
      });
    }
    return out;
  }
}

/** Records `list` `n` times into `pass`; `arg` goes to the lists whose grid depends on it. */
export function encode(pass, list, n = 1, arg = 0) {
  let last = null;
  for (let s = 0; s < n; s++) {
    for (let i = 0; i < list.length; i++) {
      const e = list[i];
      if (e.pipeline !== last) { pass.setPipeline(e.pipeline); last = e.pipeline; }
      pass.setBindGroup(0, e.bg);
      if (e.fn) {
        const [x, y = 1, z = 1] = e.fn(arg);
        pass.dispatchWorkgroups(x, y, z);
      } else pass.dispatchWorkgroups(e.x, e.y, e.z);
    }
  }
}
