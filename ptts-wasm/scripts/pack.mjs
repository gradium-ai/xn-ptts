// Assemble the `phonon-tts` npm package in `<out>` (default `pkg/`), around the wasm-pack
// output already in `<out>/wasm/`.
//
// The version is not in `js/package.json`: it is stamped here from the workspace
// `Cargo.toml`, so the npm package, the PyPI wheel and the crate cannot drift apart.
//
// Usage: node scripts/pack.mjs [out]

import { copyFileSync, existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const crateDir = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoDir = resolve(crateDir, '..');
const out = resolve(crateDir, process.argv[2] ?? 'pkg');
const js = join(crateDir, 'js');

if (!existsSync(join(out, 'wasm', 'phonon_tts_bg.wasm'))) {
  console.error(`no wasm build in ${join(out, 'wasm')}; run \`make build\` rather than this script`);
  process.exit(1);
}

const cargo = readFileSync(join(repoDir, 'Cargo.toml'), 'utf8');
const version = cargo.match(/\[workspace\.package\][^[]*?\nversion\s*=\s*"([^"]+)"/)?.[1];
if (!version) {
  console.error('could not find workspace.package.version in Cargo.toml');
  process.exit(1);
}

const manifest = JSON.parse(readFileSync(join(js, 'package.json'), 'utf8'));
// `version` right after `name`, where npm and every reader expects it.
const { name, ...rest } = manifest;
writeFileSync(join(out, 'package.json'), JSON.stringify({ name, version, ...rest }, null, 2) + '\n');

for (const file of ['index.js', 'index.d.ts', 'worker.js', 'fetch.js', 'models.js', 'wav.js', 'README.md']) {
  copyFileSync(join(js, file), join(out, file));
}
for (const file of ['LICENSE-MIT', 'LICENSE-APACHE']) {
  copyFileSync(join(repoDir, file), join(out, file));
}

// wasm-pack drops a `*` .gitignore into its output. npm reads a .gitignore in a
// subdirectory as that directory's .npmignore, and one there overrides `files`: left in
// place, it would publish a package with no wasm in it.
rmSync(join(out, 'wasm', '.gitignore'), { force: true });

console.log(`phonon-tts@${version} assembled in ${out}`);
