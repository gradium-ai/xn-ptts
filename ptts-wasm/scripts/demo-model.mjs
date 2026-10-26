#!/usr/bin/env node
// Describe a model folder as the `{ model, sizes }` the demo page loads, on stdout.
//
//   node scripts/demo-model.mjs site/model > site/model.json
//
// A static server cannot list a directory, so the page cannot find the folder's weights and
// voices by itself. This looks for the file names the Rust examples accept, and writes URLs
// relative to the page, under the folder's own name.

import { existsSync, readdirSync, statSync } from 'node:fs';
import { basename, join } from 'node:path';

const dir = process.argv[2];
if (!dir || !existsSync(join(dir, 'tokenizer.json')) || !existsSync(join(dir, 'config.json'))) {
  console.error('usage: demo-model.mjs <model folder holding config.json and tokenizer.json>');
  process.exit(1);
}
const prefix = basename(dir);
const url = (file) => `${prefix}/${file}`;

const weights = {};
const sizes = {};
for (const [quant, file] of [['q8', 'model.q8.gguf'], ['f32', 'model.safetensors']]) {
  if (existsSync(join(dir, file))) {
    weights[quant] = url(file);
    sizes[quant] = statSync(join(dir, file)).size;
  }
}
if (Object.keys(weights).length === 0) {
  console.error(`no model.q8.gguf or model.safetensors in ${dir}`);
  process.exit(1);
}

// Both voice layouts are in circulation, as in `VOICE_DIRS` in `ptts/src/loader.rs`.
const voices = {};
for (const sub of ['voices', 'embeddings']) {
  if (!existsSync(join(dir, sub))) continue;
  for (const file of readdirSync(join(dir, sub)).sort()) {
    if (file.endsWith('.safetensors')) voices[file.slice(0, -'.safetensors'.length)] ??= url(`${sub}/${file}`);
  }
}
if (existsSync(join(dir, 'default-voice.safetensors'))) voices.default ??= url('default-voice.safetensors');

const model = {
  weights,
  tokenizer: url('tokenizer.json'),
  config: url('config.json'),
  voices,
  ...('default' in voices ? { defaultVoice: 'default' } : {}),
};
console.log(JSON.stringify({ model, sizes }, null, 2));
