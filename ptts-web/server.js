// Static server for the demo: pkg/ at /, plus the local model and voices.
// WebGPU needs a secure context, and localhost counts as one, so no TLS here.
const http = require('http');
const https = require('https');
const fs = require('fs');
const os = require('os');
const path = require('path');

const PORT = process.env.PORT || 8788;
const TLS_PORT = process.env.TLS_PORT || 8789;
const PKG = path.join(__dirname, 'pkg');
const MODEL = process.env.PTTS_MODEL_DIR ||
  path.join(__dirname, '..', '..', 'phonon-inference', 'model');
const VOICES = process.env.PTTS_VOICE_DIR ||
  path.join(__dirname, '..', '..', 'phonon-inference', 'voices');

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm', '.json': 'application/json',
  '.safetensors': 'application/octet-stream', '.model': 'application/octet-stream',
  '.gguf': 'application/octet-stream', '.map': 'application/json',
};

// Only the weights are worth caching. Caching the app shell means every rebuild
// is invisible to a browser that already has the page, which looks exactly like a
// change that did not work.
function cacheControl(file) {
  const inWeights = file.startsWith(MODEL) || file.startsWith(VOICES);
  return inWeights ? 'public, max-age=86400' : 'no-cache';
}

function resolve(urlPath) {
  // Strip the query before routing: a '/?mode=x' that still carries its query
  // compares unequal to '/' and silently falls through to a directory read.
  const clean = decodeURIComponent(urlPath.split('?')[0].split('#')[0]);
  if (clean === '/' || clean === '') return path.join(PKG, 'index.html');
  // No '..' may escape the three roots we intend to expose.
  const safe = path.normalize(clean).replace(/^(\.\.[/\\])+/, '');
  if (safe.startsWith('/model/')) return path.join(MODEL, safe.slice('/model/'.length));
  if (safe.startsWith('/voices/')) return path.join(VOICES, safe.slice('/voices/'.length));
  return path.join(PKG, safe);
}

function handler(req, res) {
  // A headless run POSTs its result here; printing it and exiting is what makes
  // the browser run usable as a check from a shell.
  if (req.method === 'POST' && req.url.split('?')[0] === '/progress') {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      res.writeHead(204); res.end();
      console.log(`${new Date().toISOString().slice(11, 23)} [progress] ${body}`);
    });
    return;
  }
  if (req.method === 'POST' && req.url.split('?')[0] === '/report') {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      res.writeHead(200, { 'content-type': 'text/plain' });
      res.end('ok\n');
      console.log(body);
      if (process.env.PTTS_EXIT_ON_REPORT) {
        setTimeout(() => process.exit(body.includes('"error"') ? 1 : 0), 100);
      }
    });
    return;
  }
  const file = resolve(req.url);
  fs.stat(file, (err, st) => {
    if (process.env.PTTS_LOG_REQUESTS) {
      const t = new Date().toISOString().slice(11, 23);
      console.log(`${t} [req] ${req.url}${err || !st.isFile() ? ' (404)' : ''}`);
    }
    if (err || !st.isFile()) {
      res.writeHead(404, { 'content-type': 'text/plain' });
      return res.end(`not found: ${req.url}\n`);
    }
    res.writeHead(200, {
      'content-type': MIME[path.extname(file)] || 'application/octet-stream',
      'content-length': st.size,
      'cache-control': cacheControl(file),
    });
    fs.createReadStream(file).pipe(res);
  });
}

// Split by address range, because they are not interchangeable and printing them
// as a plain list is actively misleading: 100.64/10 is Tailscale's CGNAT range,
// and a device without Tailscale has no route to it. It does not get refused, it
// hangs, which looks exactly like the server being down.
function addresses() {
  const all = Object.values(os.networkInterfaces())
    .flat()
    .filter(i => i && i.family === 'IPv4' && !i.internal)
    .map(i => i.address);
  const isTailnet = a => {
    const [x, y] = a.split('.').map(Number);
    return x === 100 && y >= 64 && y <= 127;
  };
  return { lan: all.filter(a => !isTailnet(a)), tailnet: all.filter(isTailnet) };
}

// http on localhost, which is a secure context by definition and is what the
// local tooling talks to.
http.createServer(handler).listen(PORT, '127.0.0.1', () => {
  console.log(`http://127.0.0.1:${PORT}`);
  console.log(`  /model  -> ${MODEL}`);
  console.log(`  /voices -> ${VOICES}`);
});

// https on every interface, for phones. WebGPU needs a secure context, and a LAN
// address over plain http is not one -- `navigator.gpu` would be undefined and
// nothing on the page could run. Run scripts/make-cert.sh to create the cert.
const KEY = path.join(__dirname, 'certs', 'dev.key');
const CRT = path.join(__dirname, 'certs', 'dev.crt');
if (fs.existsSync(KEY) && fs.existsSync(CRT)) {
  const opts = { key: fs.readFileSync(KEY), cert: fs.readFileSync(CRT) };
  https.createServer(opts, handler).listen(TLS_PORT, '0.0.0.0', () => {
    const { lan, tailnet } = addresses();
    console.log(`\nOn another device on the same wifi (accept the cert warning once):`);
    for (const a of lan) console.log(`  https://${a}:${TLS_PORT}`);
    if (!lan.length) console.log('  (no LAN address found -- is wifi up?)');
    for (const a of tailnet) {
      console.log(`\nTailnet only, needs Tailscale on the other device too:`);
      console.log(`  https://${a}:${TLS_PORT}`);
    }
    console.log(
      `\nIf a browser hangs instead of erroring: macOS stealth mode drops packets\n` +
      `to ports with nothing listening, so a dead server looks like a slow one.\n` +
      `Check this process is still up, and that wifi client isolation is off.`,
    );
  });
} else {
  console.log(`\nNo cert: run scripts/make-cert.sh to enable https for phones.`);
}
