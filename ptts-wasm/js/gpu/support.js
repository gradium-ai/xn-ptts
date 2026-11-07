// What the WebGPU engine can run, decided before anything is loaded onto the device. Kept free
// of browser globals so the node tests cover it.
//
// The kernels are written for the Phonon checkpoints Gradium publishes (`gradium/phonon-31m-en`
// and the `gradium/phonon-40m-*` models): a 512-wide flow LM of any depth with an LSD flow head
// taking one step, and the 24 kHz Mimi decoder they share. A config outside that is refused
// with the first reason found, and the CPU path, which runs every config, takes it instead.

/** The `mimi` config every supported checkpoint has (`decoder_dimension` must be absent). */
const MIMI = {
  sample_rate: 24000, frame_rate: 12.5, dimension: 512, quantizer_dimension: 32, quantizer_output_dimension: 512,
  n_filters: 64, n_residual_layers: 1, kernel_size: 7, last_kernel_size: 3, residual_kernel_size: 3,
  dilation_base: 2, compress: 2, transformer_d_model: 512, transformer_num_heads: 8, transformer_num_layers: 2,
  transformer_context: 250, transformer_max_period: 10000, transformer_dim_feedforward: 2048,
};

/**
 * Why the WebGPU engine cannot run `config` (a parsed `config.json`), or null when it can.
 * @param {any} config
 * @returns {string | null}
 */
export function unsupportedConfig(config) {
  const fl = config?.flow_lm;
  const m = config?.mimi;
  if (!fl || !m) return 'the config has no flow_lm or mimi section';
  // `TTSConfig::default_gelu`: an audiocraft export (with a model_id) uses the exact GELU, a
  // pocket-tts one the tanh form, unless the config says.
  const defaultGelu = config.model_id ? 'erf' : 'tanh';
  const checks = [
    [(fl.gelu ?? defaultGelu) !== 'erf', `the flow LM's ${fl.gelu ?? defaultGelu} GELU (only the exact, erf, GELU is written)`],
    [(m.gelu ?? defaultGelu) !== 'erf', `the Mimi decoder's ${m.gelu ?? defaultGelu} GELU (only the exact, erf, GELU is written)`],
    [(fl.flow_type ?? 'lsd') !== 'lsd', `the ${fl.flow_type} flow head (only LSD is written)`],
    [(config.lsd_decode_steps ?? 1) !== 1, `lsd_decode_steps ${config.lsd_decode_steps} (only one solver step is written)`],
    [fl.d_model !== 512, `a flow LM of width ${fl.d_model} (512 is written)`],
    [fl.num_heads !== 8, `${fl.num_heads} attention heads (8 are written)`],
    [fl.num_kv_heads !== undefined && fl.num_kv_heads !== fl.num_heads, `grouped-query attention (num_kv_heads ${fl.num_kv_heads})`],
    [fl.attn_dim !== undefined && fl.attn_dim !== fl.d_model, `attn_dim ${fl.attn_dim}`],
    [fl.z_dim !== undefined, `a z_proj (z_dim ${fl.z_dim})`],
    [fl.dim_feedforward !== 2048, `a feed-forward width of ${fl.dim_feedforward} (2048 is written)`],
    [fl.flow_dim !== 512, `a flow net of width ${fl.flow_dim} (512 is written)`],
    [fl.ldim !== 32, `${fl.ldim}-wide latents (32 are written)`],
    [fl.lut_dim !== fl.d_model, `a low-rank text table (lut_dim ${fl.lut_dim})`],
    [fl.max_period !== 10000, `a RoPE period of ${fl.max_period} (10000 is written)`],
    [!(fl.num_layers >= 1), `${fl.num_layers} flow-LM layers`],
    [!(fl.flow_depth >= 1), `a flow net of depth ${fl.flow_depth}`],
    [(config.prefix_sinks?.length ?? 0) > 0, 'prefix sinks'],
    [(config.audio_delay ?? 0) !== 0, `an audio delay of ${config.audio_delay} frames`],
    [(config.voices?.length ?? 0) > 0, 'baked-in voices'],
    [m.decoder_dimension !== undefined && m.decoder_dimension !== m.dimension, `a Mimi decoder of width ${m.decoder_dimension}`],
    [JSON.stringify(m.ratios) !== '[6,5,4]', `Mimi ratios ${JSON.stringify(m.ratios)} ([6, 5, 4] are written)`],
    [!(m.transformer_layer_scale > 0), 'a Mimi transformer without layer scale'],
  ];
  for (const [bad, what] of checks) if (bad) return `the WebGPU engine does not run ${what}`;
  for (const [k, v] of Object.entries(MIMI)) {
    if (m[k] !== v) return `the WebGPU engine does not run this Mimi decoder (${k} ${m[k]}, ${v} is written)`;
  }
  return null;
}

/** ggml type of q8_0. */
const Q8_0 = 8;

/**
 * Why the engine cannot read these weights (a parsed GGUF), or null. The transformer's linears
 * must be q8_0, as `quantize` writes them; `model.q4k.gguf` and the like are refused.
 */
export function unsupportedWeights(g, config) {
  for (let l = 0; l < config.flow_lm.num_layers; l++) {
    for (const w of ['self_attn.in_proj', 'self_attn.out_proj', 'linear1', 'linear2']) {
      const name = `flow_lm.transformer.layers.${l}.${w}.weight`;
      const t = g.tensors.get(name);
      if (!t) return `the weights have no tensor ${name}`;
      if (t.type !== Q8_0) return `the WebGPU engine reads q8_0 weights, and ${name} has ggml type ${t.type}`;
    }
  }
  return null;
}

/**
 * Which reductions the kernels use on this adapter. The subgroup kernels sum across a 32-lane
 * subgroup and take `local_invocation_index >> 5` to name it, so they need subgroups exactly
 * 32 wide; anything else, or no subgroups at all, gets the shared-memory kernels.
 * @param {{ features: { has(name: string): boolean }, info?: { subgroupMinSize?: number, subgroupMaxSize?: number } }} adapter
 * @returns {{ subgroups: boolean, reason: string }}
 */
export function chooseReductions(adapter) {
  if (!adapter.features.has('subgroups')) return { subgroups: false, reason: 'the adapter has no subgroups' };
  const { subgroupMinSize: lo, subgroupMaxSize: hi } = adapter.info ?? {};
  if (lo !== 32 || hi !== 32) return { subgroups: false, reason: `the adapter's subgroups are ${lo ?? '?'} to ${hi ?? '?'} wide, not 32` };
  return { subgroups: true, reason: '32-wide subgroups' };
}

/** A u64 seed (a bigint, as the package takes it) as the 32 bits the GPU's noise hash uses. */
export function seed32(seed) {
  const s = BigInt.asUintN(64, BigInt(seed));
  return Number(BigInt.asUintN(32, s ^ (s >> 32n)));
}
