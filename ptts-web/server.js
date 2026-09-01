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
// The ASR checkpoint, as huggingface-cli left it. Resolved through the snapshot
// symlinks so the served paths are stable across re-downloads.
const ASR = process.env.PTTS_ASR_DIR || (() => {
  const base = path.join(process.env.HOME || '', '.cache', 'huggingface', 'hub',
    'models--gr4d--asr-23b5a198.500', 'snapshots');
  try {
    const snap = fs.readdirSync(base).map(d => path.join(base, d))
      .find(d => fs.existsSync(path.join(d, 'config.json')));
    if (snap) return snap;
  } catch { /* not downloaded; /asr just 404s */ }
  return path.join(base, 'missing');
})();

const OPENROUTER_KEY = process.env.OPENROUTER_API_KEY || '';
const LLM_MODEL = process.env.PTTS_LLM_MODEL || 'liquid/lfm-2.5-2.6b:free';

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.wasm': 'application/wasm', '.json': 'application/json',
  '.safetensors': 'application/octet-stream', '.model': 'application/octet-stream',
  '.wav': 'audio/wav',
  '.gguf': 'application/octet-stream', '.map': 'application/json',
};

// Only the weights are worth caching. Caching the app shell means every rebuild
// is invisible to a browser that already has the page, which looks exactly like a
// change that did not work.
function cacheControl(file) {
  const inWeights = file.startsWith(MODEL) || file.startsWith(VOICES) || file.startsWith(ASR);
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
  if (safe.startsWith('/asr/')) return path.join(ASR, safe.slice('/asr/'.length));
  // Sample audio, so the ASR can be driven from a file where there is no mic.
  if (safe.startsWith('/audio/')) return path.join(MODEL, '..', safe.slice('/audio/'.length));
  return path.join(PKG, safe);
}

function handler(req, res) {
  // A headless run POSTs its result here; printing it and exiting is what makes
  // the browser run usable as a check from a shell.
  // The LLM leg. Proxied rather than called from the page so the OpenRouter key
  // stays on this machine and never reaches the browser.
  if (req.method === 'POST' && req.url.split('?')[0] === '/api/chat') {
    let body = '';
    req.on('data', c => { body += c; });
    req.on('end', async () => {
      const json = h => { res.writeHead(h.code, { 'content-type': 'application/json' }); res.end(JSON.stringify(h.body)); };
      if (!OPENROUTER_KEY) {
        return json({ code: 500, body: { error: 'OPENROUTER_API_KEY is not set in this server process' } });
      }
      let messages;
      try { ({ messages } = JSON.parse(body)); } catch { return json({ code: 400, body: { error: 'bad json' } }); }
      if (!Array.isArray(messages)) return json({ code: 400, body: { error: 'messages must be an array' } });

      // The free tier shares an upstream pool and 429s regularly, with a
      // Retry-After that is worth honouring rather than failing the turn.
      for (let attempt = 0; attempt < 3; attempt++) {
        let r, data;
        try {
          r = await fetch('https://openrouter.ai/api/v1/chat/completions', {
            method: 'POST',
            headers: { 'authorization': `Bearer ${OPENROUTER_KEY}`, 'content-type': 'application/json' },
            body: JSON.stringify({ model: LLM_MODEL, messages, max_tokens: 160, temperature: 0.7 }),
          });
          data = await r.json();
        } catch (e) {
          return json({ code: 502, body: { error: `openrouter unreachable: ${e.message}` } });
        }
        const text = data?.choices?.[0]?.message?.content;
        if (text) return json({ code: 200, body: { text, model: data.model || LLM_MODEL } });
        const retryAfter = data?.error?.metadata?.retry_after_seconds;
        const rateLimited = data?.error?.code === 429 || r.status === 429;
        if (rateLimited && attempt < 2) {
          const wait = Math.min(20, retryAfter || 5);
          console.log(`[llm] ${LLM_MODEL} rate limited upstream, retrying in ${wait}s (attempt ${attempt + 1}/3)`);
          await new Promise(z => setTimeout(z, wait * 1000));
          continue;
        }
        return json({ code: 502, body: { error: data?.error?.message || 'no completion returned', detail: data?.error || null } });
      }
    });
    return;
  }
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
