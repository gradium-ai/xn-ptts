// Static server for the demo: pkg/ at /, plus the local model and voices.
// WebGPU needs a secure context, and localhost counts as one, so no TLS here.
const http = require('http');
const fs = require('fs');
const path = require('path');

const PORT = process.env.PORT || 8788;
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

http.createServer((req, res) => {
  // A headless run POSTs its result here; printing it and exiting is what makes
  // the browser run usable as a check from a shell.
  if (req.method === 'POST' && req.url.split('?')[0] === '/progress') {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', () => {
      res.writeHead(204); res.end();
      console.log('[progress] ' + body);
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
      console.log(`[req] ${req.url} -> ${file}${err || !st.isFile() ? ' (404)' : ''}`);
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
}).listen(PORT, '127.0.0.1', () => {
  console.log(`http://127.0.0.1:${PORT}`);
  console.log(`  /model  -> ${MODEL}`);
  console.log(`  /voices -> ${VOICES}`);
});
