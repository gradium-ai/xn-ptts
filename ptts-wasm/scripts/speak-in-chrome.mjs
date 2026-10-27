#!/usr/bin/env node
// Speak a line through the built package in headless Chrome and save it as a WAV file, so CI
// can check the browser build end to end: the worker, the wasm, the Hub download and the device.
//
//   node scripts/speak-in-chrome.mjs --pkg pkg --text "Hello." --out hello.wav \
//     [--device cpu|webgpu] [--no-isolation] [--chrome PATH] [-- CHROME_FLAGS...]
//
// The page loads the package's default model. It is served cross-origin isolated, so it runs
// the threaded build, unless --no-isolation asks for the single-threaded one. Flags after `--`
// go to Chrome, e.g. what a machine without a GPU needs for WebGPU.

import { spawn } from 'node:child_process';
import { createReadStream, mkdtempSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { extname, join, normalize, resolve, sep } from 'node:path';
import { parseArgs } from 'node:util';

const split = process.argv.indexOf('--');
const { values: args } = parseArgs({
  args: process.argv.slice(2, split < 0 ? undefined : split),
  options: {
    pkg: { type: 'string', default: 'pkg' },
    text: { type: 'string' },
    out: { type: 'string' },
    device: { type: 'string', default: 'cpu' },
    'no-isolation': { type: 'boolean', default: false },
    chrome: { type: 'string' },
    timeout: { type: 'string', default: '600' },
  },
});
if (!args.text || !args.out) {
  console.error('usage: speak-in-chrome.mjs --pkg DIR --text TEXT --out FILE.wav [--device cpu|webgpu]');
  process.exit(2);
}
const chromeFlags = split < 0 ? [] : process.argv.slice(split + 1);
const chrome = args.chrome ?? process.env.CHROME_BIN ?? (process.platform === 'darwin'
  ? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
  : 'google-chrome');

const PAGE = `<!doctype html><meta charset="utf-8"><script type="module">
const report = (path, body) => fetch(path, { method: 'POST', body });
try {
  const { PhononTTS } = await import('./pkg/index.js');
  const tts = await PhononTTS.load({ lang: 'en', device: ${JSON.stringify(args.device)} });
  const wav = await tts.synthWav(${JSON.stringify(args.text)});
  await report('/info', JSON.stringify({ device: tts.device, threads: tts.threads, features: tts.features }));
  await report('/wav', wav);
} catch (e) {
  await report('/error', String(e?.stack ?? e));
}
</script>`;

const TYPES = { '.js': 'text/javascript', '.wasm': 'application/wasm', '.json': 'application/json' };
const pkg = resolve(args.pkg);
let info = null;

const finished = new Promise((done, fail) => {
  const server = createServer(async (req, res) => {
    if (!args['no-isolation']) {
      res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
      res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    }
    const path = new URL(req.url, 'http://localhost').pathname;
    if (req.method === 'POST') {
      const chunks = [];
      for await (const chunk of req) chunks.push(chunk);
      res.end();
      const body = Buffer.concat(chunks);
      if (path === '/info') info = JSON.parse(body);
      else if (path === '/wav') done(body);
      else fail(new Error(`the page failed: ${body}`));
      return;
    }
    if (path === '/') return res.writeHead(200, { 'Content-Type': 'text/html' }).end(PAGE);
    try {
      const file = normalize(join(pkg, decodeURIComponent(path.replace(/^\/pkg\//, '/'))));
      if (!path.startsWith('/pkg/') || !file.startsWith(pkg + sep)) throw new Error('outside');
      const { size } = statSync(file);
      res.writeHead(200, { 'Content-Type': TYPES[extname(file)] ?? 'application/octet-stream', 'Content-Length': size });
      createReadStream(file).pipe(res);
    } catch {
      res.writeHead(404).end();
    }
  });
  server.listen(0, '127.0.0.1', () => {
    const profile = mkdtempSync(join(tmpdir(), 'speak-in-chrome-'));
    const url = `http://127.0.0.1:${server.address().port}/`;
    const browser = spawn(chrome, [
      '--headless=new', '--no-first-run', '--no-default-browser-check', `--user-data-dir=${profile}`,
      '--enable-logging=stderr', '--v=0', ...chromeFlags, url,
    ], { stdio: ['ignore', 'ignore', 'pipe'] });
    let log = '';
    browser.stderr.on('data', (d) => { log += d; });
    browser.on('error', fail);
    browser.on('exit', (code) => fail(new Error(`Chrome exited with ${code}:\n${log}`)));
    const timer = setTimeout(() => fail(new Error(`no audio after ${args.timeout} s:\n${log}`)), args.timeout * 1000);
    finished.finally(() => {
      clearTimeout(timer);
      browser.removeAllListeners('exit');
      browser.kill();
      server.close();
      server.closeAllConnections();
      try { rmSync(profile, { recursive: true, force: true }); } catch {}
    }).catch(() => {});
  });
});

try {
  const wav = await finished;
  writeFileSync(args.out, wav);
  console.log(`${args.out}: ${wav.length} bytes on ${info.device}, ${info.threads} thread(s), ${JSON.stringify(info.features)}`);
  if (info.device !== args.device) throw new Error(`asked for ${args.device}, ran on ${info.device}`);
} catch (e) {
  console.error(e.message);
  process.exit(1);
}
