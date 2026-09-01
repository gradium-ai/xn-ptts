// Drives the page over CDP and captures composited screenshots while it runs.
// Screenshot diffs are the only thing that proves the waveform reaches the
// screen; reading the canvas back would pass even if it never painted.
import { createHash } from 'node:crypto';

const [, , wsUrl, pageUrl] = process.argv;
const ws = new WebSocket(wsUrl);
let id = 0;
const pending = new Map();

const send = (method, params = {}) =>
  new Promise((res, rej) => {
    const msgId = ++id;
    pending.set(msgId, { res, rej });
    ws.send(JSON.stringify({ id: msgId, method, params }));
  });

ws.onmessage = ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id);
    pending.delete(m.id);
    m.error ? rej(new Error(m.error.message)) : res(m.result);
  }
};

const sleep = ms => new Promise(r => setTimeout(r, ms));

ws.onopen = async () => {
  await send('Page.enable');
  await send('Runtime.enable');
  await send('Page.navigate', { url: pageUrl });

  const status = async () => {
    try {
      const r = await send('Runtime.evaluate', {
        expression: "document.getElementById('status').textContent",
        returnByValue: true,
      });
      return r.result.value || '';
    } catch { return ''; }
  };

  const samples = [];
  const deadline = Date.now() + 240000;
  let sawDone = 0;
  while (Date.now() < deadline) {
    const st = await status();
    if (/^generating/.test(st)) {
      const shot = await send('Page.captureScreenshot', { format: 'png' });
      samples.push({
        st: st.slice(0, 40),
        hash: createHash('sha1').update(shot.data).digest('hex').slice(0, 10),
        bytes: shot.data.length,
      });
    }
    if (/^done/.test(st)) { sawDone++; if (sawDone > 6) break; }
    await sleep(100);
  }

  const distinct = new Set(samples.map(s => s.hash)).size;
  console.log(`screenshots while generating: ${samples.length}, distinct: ${distinct}`);
  for (const s of samples.slice(0, 14)) console.log(`  ${s.hash}  ${s.bytes} B  ${s.st}`);
  console.log(distinct >= 3
    ? `VISIBLY CHANGING (${distinct} distinct composited frames during generation)`
    : 'NOT CHANGING ON SCREEN');
  ws.close();
  process.exit(0);
};
ws.onerror = e => { console.error('cdp error', e.message); process.exit(1); };
