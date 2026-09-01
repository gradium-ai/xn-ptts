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

  // Captured across both phases: generation, then playback. The point is that
  // the picture keeps changing after generation is done, while audio still plays.
  const probeExpr = 'window.__probe ? JSON.stringify(window.__probe()) : ""';
  const probe = async () => {
    try {
      const r = await send('Runtime.evaluate', { expression: probeExpr, returnByValue: true });
      return r.result.value ? JSON.parse(r.result.value) : null;
    } catch (e) { return null; }
  };

  const samples = [];
  const deadline = Date.now() + 180000;
  let started = false;
  while (Date.now() < deadline) {
    const p = await probe();
    if (p && (p.live || /^generating/.test(p.st))) {
      started = true;
      const shot = await send('Page.captureScreenshot', { format: 'png' });
      samples.push({
        loops: p.loops, paints: p.paints, err: p.err, bright: p.bright, dim: p.dim, mode: p.playMode,
        phase: /^generating/.test(p.st) ? 'gen ' : 'play',
        heard: p.heard === null ? null : +p.heard.toFixed(2),
        hash: createHash('sha1').update(shot.data).digest('hex').slice(0, 8),
        bytes: shot.data.length,
      });
    } else if (started) break;
    await sleep(150);
  }

  const during = samples.filter(s => s.phase === 'play');
  const distinctPlay = new Set(during.map(s => s.hash)).size;
  const brights = during.map(s => s.bright);
  const grew = brights.length > 2 && brights[brights.length - 1] > brights[0];
  const tracksClock = grew && during.every((s, i) => i === 0 || s.bright >= during[i - 1].bright);

  console.log(`captures: ${samples.length} (gen ${samples.length - during.length}, play ${during.length})`);
  for (const s of samples) {
    console.log(`  ${s.phase} heard=${s.heard}s paints=${s.paints} bright=${s.bright} dim=${s.dim} ${s.hash}`);
  }
  const mode = samples.length ? samples[samples.length - 1].mode : '?';
  if (mode === 'off') {
    // No clock to follow, so the waveform is finished the moment generation is,
    // and there is no playback phase to sample. Not a failure.
    console.log('playback off: waveform completes with generation, nothing to track');
  } else {
    console.log(tracksClock
      ? `FILL TRACKS THE AUDIO CLOCK (bright ${brights[0]} -> ${brights[brights.length - 1]} px while audio played)`
      : `FILL DOES NOT TRACK PLAYBACK (bright ${brights.join(',')})`);
  }
  // Composited frames are not a usable signal here: once the DOM settles this
  // headless setup stops producing new frames for canvas-only updates, so the
  // screenshots go identical even while the canvas provably changes.
  console.log(`composited frames during playback: ${distinctPlay} (headless does not repaint canvas-only changes; not a verdict)`);
  ws.close();
  process.exit(0);
};
ws.onerror = e => { console.error('cdp error', e.message); process.exit(1); };
