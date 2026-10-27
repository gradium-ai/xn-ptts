// Assemble the `phonon-tts` npm package in `<out>` (default `pkg/`), around the wasm-pack
// output already in `<out>/wasm/`, and in `<out>/wasm-threads/` when the threaded build ran.
//
// The version is not in `js/package.json`: it is stamped here from the workspace
// `Cargo.toml`, so the npm package, the PyPI wheel and the crate cannot drift apart.
//
// Usage: node scripts/pack.mjs [out]

import { copyFileSync, existsSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
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
// Only inside `[workspace.package]`, up to the next section header, so an array value such as
// `keywords = [...]` in between cannot end the search early.
const section = cargo.match(/^\[workspace\.package\]\s*$([\s\S]*?)(?=^\[|(?![\s\S]))/m)?.[1] ?? '';
const version = section.match(/^version\s*=\s*"([^"]+)"/m)?.[1];
if (!version) {
  console.error('could not find workspace.package.version in Cargo.toml');
  process.exit(1);
}

const manifest = JSON.parse(readFileSync(join(js, 'package.json'), 'utf8'));
// `version` right after `name`, where npm and every reader expects it.
const { name, ...rest } = manifest;
writeFileSync(join(out, 'package.json'), JSON.stringify({ name, version, ...rest }, null, 2) + '\n');

// Everything `files` names outside the two wasm builds and the licences comes from js/, so
// the copy list and the publish list cannot drift apart. A module added to one but not the
// other would otherwise ship a package missing a file, and nothing here would say so.
const fromJs = manifest.files.filter(
  (f) =>
    !f.startsWith('wasm/') &&
    !f.startsWith('wasm-threads/') &&
    !f.startsWith('LICENSE') &&
    f !== 'THIRD_PARTY_LICENSES.md',
);
for (const file of [...fromJs, 'README.md']) {
  copyFileSync(join(js, file), join(out, file));
}
for (const file of ['LICENSE-MIT', 'LICENSE-APACHE']) {
  copyFileSync(join(repoDir, file), join(out, file));
}

// `make build` writes the third-party notices with cargo-about. npm leaves out a `files` entry
// that does not exist, so a package without them would publish without a word.
if (!existsSync(join(out, 'THIRD_PARTY_LICENSES.md'))) {
  console.warn(`no THIRD_PARTY_LICENSES.md in ${out}: this package must not be published`);
}

// wasm-pack drops a `*` .gitignore into its output. npm reads a .gitignore in a
// subdirectory as that directory's .npmignore, and one there overrides `files`: left in
// place, it would publish a package with no wasm in it.
rmSync(join(out, 'wasm', '.gitignore'), { force: true });

// The threaded build, when there is one. `make profiling` builds without it, and the worker
// then runs single threaded.
const threads = join(out, 'wasm-threads');
if (existsSync(join(threads, 'phonon_tts_bg.wasm'))) {
  rmSync(join(threads, '.gitignore'), { force: true });
  // wasm-bindgen-rayon's worker helper imports the module it belongs to as `'../../..'`,
  // which a bundler resolves through a package.json and a browser resolves to a directory,
  // so it works for neither here. Point it at the module file itself.
  const snippets = join(threads, 'snippets');
  const helpers = readdirSync(snippets)
    .filter((d) => d.startsWith('wasm-bindgen-rayon-'))
    .map((d) => join(snippets, d, 'src', 'workerHelpers.js'));
  if (helpers.length !== 1) {
    console.error(`expected one wasm-bindgen-rayon worker helper in ${snippets}, found ${helpers.length}`);
    process.exit(1);
  }
  const before = readFileSync(helpers[0], 'utf8');
  const after = before.replace("import('../../..')", "import('../../../phonon_tts.js')");
  if (!after.includes("import('../../../phonon_tts.js')")) {
    console.error(`${helpers[0]} no longer imports '../../..': check wasm-bindgen-rayon's worker helper`);
    process.exit(1);
  }
  writeFileSync(helpers[0], after);
}

// The builds without Relaxed SIMD, for browsers that lack it, when they ran. Only their
// module is kept, beside the Relaxed one: the worker loads it through the same JS, which
// only works while wasm-bindgen generates the same JS for both. Checked here rather than
// assumed. Comment lines are left out of the comparison: wasm-bindgen notes internal
// indices in them that differ between builds.
for (const [build, simd] of [['wasm', 'wasm-simd'], ['wasm-threads', 'wasm-threads-simd']]) {
  const from = join(out, simd);
  if (!existsSync(join(from, 'phonon_tts_bg.wasm'))) {
    console.warn(`no ${simd} build: browsers without Relaxed SIMD cannot load ${build}/`);
    continue;
  }
  const glue = (dir) => readFileSync(join(dir, 'phonon_tts.js'), 'utf8').replace(/^\s*\/\/.*$/gm, '');
  if (glue(from) !== glue(join(out, build))) {
    console.error(`${simd}/phonon_tts.js differs from ${build}/phonon_tts.js: the two modules need the same JS`);
    process.exit(1);
  }
  copyFileSync(join(from, 'phonon_tts_bg.wasm'), join(out, build, 'phonon_tts_simd_bg.wasm'));
  rmSync(from, { recursive: true, force: true });
}

console.log(`phonon-tts@${version} assembled in ${out}`);
