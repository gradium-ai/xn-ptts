// Attaches to every target as it starts -- including the worker the model runs
// in -- and patches the WebGPU entry points to count real calls. Counting is the
// only way to tell "ships a webgpu backend" from "actually ran on the gpu".
const [, , browserWs, pageUrl] = process.argv;
const ws = new WebSocket(browserWs);
let id = 0; const pending = new Map(); const sessions = new Map();
const send = (method, params = {}, sessionId) => new Promise((res, rej) => {
  const i = ++id; pending.set(i, { res, rej });
  ws.send(JSON.stringify({ id: i, method, params, ...(sessionId ? { sessionId } : {}) }));
});

// Idempotent: a target can be attached more than once (browser-level and
// page-level auto-attach both fire), and re-running a non-idempotent hook would
// zero the counters after work had already been counted.
// Instance-level, deliberately. The worker is paused before its globals are
// populated, so `GPUAdapter.prototype` and friends do not exist yet at hook time
// and patching them silently no-ops -- which reads as "the GPU was never used".
// Wrapping what `requestAdapter` hands back cannot miss: every device, queue,
// encoder and pass descends from it.
const HOOKS = `
  if (!self.__gpuHooked) {
    self.__gpuHooked = true;
    self.__gpu = { requestAdapter: 0, adapterOk: 0, requestDevice: 0, deviceOk: 0,
                   shaderModules: 0, pipelines: 0, submits: 0, dispatches: 0 };
    const c = self.__gpu;
    const g = self.navigator && self.navigator.gpu;
    if (g && g.requestAdapter) {
      const ra = g.requestAdapter.bind(g);
      g.requestAdapter = async (...a) => {
        c.requestAdapter++;
        const adapter = await ra(...a);
        if (!adapter) return adapter;
        c.adapterOk++;
        const rd = adapter.requestDevice.bind(adapter);
        adapter.requestDevice = async (...b) => {
          c.requestDevice++;
          const dev = await rd(...b);
          if (!dev) return dev;
          c.deviceOk++;
          const csm = dev.createShaderModule.bind(dev);
          dev.createShaderModule = (...x) => { c.shaderModules++; return csm(...x); };
          const ccp = dev.createComputePipeline.bind(dev);
          dev.createComputePipeline = (...x) => { c.pipelines++; return ccp(...x); };
          const sub = dev.queue.submit.bind(dev.queue);
          dev.queue.submit = (...x) => { c.submits++; return sub(...x); };
          const cce = dev.createCommandEncoder.bind(dev);
          dev.createCommandEncoder = (...x) => {
            const enc = cce(...x);
            const bcp = enc.beginComputePass.bind(enc);
            enc.beginComputePass = (...y) => {
              const pass = bcp(...y);
              const dw = pass.dispatchWorkgroups.bind(pass);
              pass.dispatchWorkgroups = (...z) => { c.dispatches++; return dw(...z); };
              return pass;
            };
            return enc;
          };
          return dev;
        };
        return adapter;
      };
    }
  }
`;

ws.onmessage = async ev => {
  const m = JSON.parse(ev.data);
  if (m.id && pending.has(m.id)) {
    const { res, rej } = pending.get(m.id); pending.delete(m.id);
    m.error ? rej(new Error(m.error.message)) : res(m.result);
    return;
  }
  if (m.method === 'Target.attachedToTarget') {
    const { sessionId, targetInfo } = m.params;
    sessions.set(sessionId, targetInfo.type);
    try {
      await send('Runtime.enable', {}, sessionId);
      // Installed before the target's own code runs, so nothing is missed.
      await send('Page.addScriptToEvaluateOnNewDocument', { source: HOOKS }, sessionId).catch(() => {});
      await send('Runtime.evaluate', { expression: HOOKS }, sessionId);
      await send('Runtime.runIfWaitingForDebugger', {}, sessionId).catch(() => {});
    } catch (e) { /* target may be gone */ }
  }
};

const sleep = ms => new Promise(r => setTimeout(r, ms));

ws.onopen = async () => {
  await send('Target.setAutoAttach', { autoAttach: true, waitForDebuggerOnStart: true, flatten: true });
  const { targetId } = await send('Target.createTarget', { url: 'about:blank' });
  const { sessionId } = await send('Target.attachToTarget', { targetId, flatten: true });
  await send('Page.enable', {}, sessionId);
  await send('Runtime.enable', {}, sessionId);
  // Dedicated workers attach under their page's session, not the browser's, and
  // the model runs in one -- without this the counters watch the wrong contexts.
  await send('Target.setAutoAttach',
    { autoAttach: true, waitForDebuggerOnStart: true, flatten: true }, sessionId);
  await send('Page.addScriptToEvaluateOnNewDocument', { source: HOOKS }, sessionId);
  await send('Page.navigate', { url: pageUrl }, sessionId);

  const evalPage = expr => send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true }, sessionId)
    .then(r => r.result.value).catch(e => 'ERR ' + e.message);

  // Wait for the worker to report support and enable the webgpu option.
  for (let i = 0; i < 60; i++) {
    const st = await evalPage("document.getElementById('webgpu-option').disabled");
    if (st === false) break;
    await sleep(1000);
  }
  console.log('webgpu option enabled:', await evalPage("!document.getElementById('webgpu-option').disabled"));
  console.log('option label       :', await evalPage("document.getElementById('webgpu-option').textContent"));

  await evalPage("document.getElementById('quant').value = 'webgpu'; document.getElementById('load-btn').click(); 'clicked'");
  console.log('selected backend   :', await evalPage("document.getElementById('quant').value"));

  // Wait on the status text, not on the button: the button is enabled before the
  // f32 safetensors (235 MB) has finished downloading, so clicking on it races
  // the load and generation never runs.
  let lastStatus = '';
  let ready = false;
  for (let i = 0; i < 400; i++) {
    const st = await evalPage("document.getElementById('status').textContent.slice(0,90)");
    if (st !== lastStatus) { console.log('  load:', st); lastStatus = st; }
    if (typeof st === 'string' && /Ready \(backend=/.test(st)) { ready = true; break; }
    await sleep(2000);
  }
  console.log('model ready       :', ready);
  const preGen = await send('Runtime.evaluate',
    { expression: 'JSON.stringify(self.__gpu)', returnByValue: true },
    [...sessions].find(([, t]) => t === 'worker')?.[0]).then(r => r.result.value).catch(() => 'n/a');
  console.log('counters at ready :', preGen);

  await evalPage("document.getElementById('generate-btn').click(); 'gen'");
  lastStatus = '';
  for (let i = 0; i < 40; i++) {
    const st = await evalPage("document.getElementById('status').textContent.slice(0,110)");
    if (st !== lastStatus) { console.log('  gen status:', st); lastStatus = st; }
    await sleep(1500);
    const done = await evalPage("!!document.querySelector('#audio-container audio')");
    if (done === true) { console.log('  audio element produced'); break; }
  }

  console.log('\\n--- WebGPU calls counted, per target ---');
  for (const [sid, type] of sessions) {
    const r = await send('Runtime.evaluate', { expression: 'JSON.stringify(self.__gpu || null)', returnByValue: true }, sid).catch(() => null);
    if (r && r.result.value && r.result.value !== 'null' && /worker|page/.test(type)) console.log(`  ${type}: ${r.result.value}`);
  }
  ws.close(); process.exit(0);
};
ws.onerror = e => { console.error('cdp error', e.message); process.exit(1); };
