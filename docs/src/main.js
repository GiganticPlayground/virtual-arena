// App entry point. With the OffscreenCanvas refactor, the main thread does
// only: DOM control wiring, <video> lifecycle, foam-finger overlay transforms,
// and per-rAF cam frame capture + postMessage. All WebGL + MediaPipe work is
// in ./worker.js.

import {
  canvas, video, fpsEl, infEl, toggleBtn,
  featherInput, featherVal, mirrorInput, foamToggle, handCountSel,
  infRateInput, infRateVal, sourceSel, uploadVideoInput,
  camSize, camSizeVal, camX, camXVal, camY, camYVal,
  modelSel, rvmQualitySel, trimInput, trimVal,
  progressSub, progressLabel,
  showError, clearError,
} from './dom.js';
import { postFrame, getLandmarks, clearLandmarks, onRender, onInfer, setActiveModel, setRvmQuality, setNumHands, onRvmStatus, canInfer } from './worker-client.js';
import { startSource, teardownSource } from './source.js';
import { applyFoam, hideFoamFrom } from './foam.js';
import { populateArenaSelect, wireArenaSelect, wireBackgroundUpload } from './backgrounds.js';
import { preload } from './preload.js';

// Browser-compat banner. The ONNX paths need JSPI (Chrome 137+) plus the
// VideoFrame API, so non-Chromium browsers will crash the worker as soon
// as the user picks an ONNX model. Warn once, dismissal is sticky via
// localStorage. navigator.userAgentData is a Chromium-only API so its
// presence is a reliable "this is Chromium" signal without UA sniffing.
(function warnIfNotChromium() {
  let dismissed = false;
  try { dismissed = localStorage.getItem('va_chrome_warn_dismissed') === '1'; } catch {}
  if (dismissed) return;
  if (navigator.userAgentData) return;

  const banner = document.createElement('div');
  banner.id = 'browser-warning';
  banner.innerHTML = `
    <span><strong>Heads up:</strong> this prototype is built for Chrome and has not been tested on other browsers. ONNX model options may fail to load.</span>
    <button type="button" aria-label="Dismiss">&times;</button>
  `;
  document.body.appendChild(banner);
  banner.querySelector('button').addEventListener('click', () => {
    try { localStorage.setItem('va_chrome_warn_dismissed', '1'); } catch {}
    banner.remove();
  });
})();

// Transfer canvas control to the worker BEFORE anything else touches it.
// After this call, the <canvas> element stays in the DOM (CSS layout still
// applies) but main can no longer getContext or draw on it.
const offscreenCanvas = canvas.transferControlToOffscreen();

// ---- UI wiring --------------------------------------------------------
populateArenaSelect();
wireArenaSelect();
wireBackgroundUpload();

for (const [input, span] of [[camSize, camSizeVal], [camX, camXVal], [camY, camYVal]]) {
  const update = () => { span.textContent = (+input.value).toFixed(2); };
  input.addEventListener('input', update);
  update();
}
infRateInput.addEventListener('input', () => { infRateVal.textContent = infRateInput.value; });
featherInput.addEventListener('input', () => { featherVal.textContent = (+featherInput.value).toFixed(1); });
trimInput.addEventListener('input',    () => { trimVal.textContent    = (+trimInput.value).toFixed(2); });
// Quality dropdown only meaningful for RVM. u2netp and silueta ship as
// fixed-shape ONNX (320x320), so changing input size throws at OrtRun.
modelSel.addEventListener('change',    () => {
  setActiveModel(modelSel.value);
  rvmQualitySel.disabled = modelSel.value !== 'rvm';
});
rvmQualitySel.addEventListener('change', () => { setRvmQuality(rvmQualitySel.value); });
rvmQualitySel.disabled = modelSel.value !== 'rvm';
handCountSel.addEventListener('change', () => { setNumHands(+handCountSel.value); });

// Surface ONNX model (first-time) load state in the status box so the
// user sees why the cutout hasn't switched yet. Subsequent toggles are
// instant (session stays in memory).
const ONNX_LOAD_LABELS = {
  rvm:     'Loading RVM model (~15 MB + ORT runtime)…',
  u2netp:  'Loading U²-Net-p model (~4 MB + ORT runtime)…',
  silueta: 'Loading Silueta model (~42 MB + ORT runtime)…',
  modnet:  'Loading MODNet model (~26 MB + ORT runtime)…',
};
onRvmStatus((m) => {
  if (m.status === 'loading')    showError(ONNX_LOAD_LABELS[m.model] || 'Loading ONNX model…');
  else if (m.status === 'ready') clearError();
});
foamToggle.addEventListener('change', () => {
  if (!foamToggle.checked) hideFoamFrom(0);
});

sourceSel.addEventListener('change', async () => {
  if (!running) return;
  toggleBtn.disabled = true;
  try { await startSource(); }
  catch (e) { showError(e.message); stop(); }
  finally { toggleBtn.disabled = false; }
});

// User-supplied video → single "Custom" entry in the Source dropdown.
let customVideoBlobUrl = null;
uploadVideoInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  e.target.value = '';
  if (!file) return;
  if (customVideoBlobUrl) URL.revokeObjectURL(customVideoBlobUrl);
  customVideoBlobUrl = URL.createObjectURL(file);
  const label = 'Custom — ' + file.name;
  let opt = Array.from(sourceSel.options).find(o => o.dataset.custom === 'true');
  if (!opt) {
    opt = document.createElement('option');
    opt.dataset.custom = 'true';
    sourceSel.appendChild(opt);
  }
  opt.value = customVideoBlobUrl;
  opt.textContent = label;
  sourceSel.value = customVideoBlobUrl;
  if (running) {
    toggleBtn.disabled = true;
    try { await startSource(); }
    catch (err) { showError(err.message); stop(); }
    finally { toggleBtn.disabled = false; }
  }
});

// ---- Start / Stop / rAF loop -----------------------------------------
let running = false;
let rafId   = 0;

// Event-driven HUD: Render FPS counts actual worker-side draws; Infer FPS
// counts ticks that carried segmentation or hand inference. Both use an
// EMA so each event updates the display immediately with a smoothed value.
// This matters for slow models like RVM (~1-2 Hz) where any render-tick
// windowing would stale out between inferences and read "—".
let renderEma = 0, lastRenderMs = -1;
let inferEma  = 0, lastInferMs  = -1;
let lowWarned = false;

function fmtFps(v) { return v.toFixed(0).padStart(3, ' '); }

onRender(() => {
  const now = performance.now();
  if (lastRenderMs > 0) {
    const inst = 1000 / Math.max(now - lastRenderMs, 1);
    renderEma = renderEma ? renderEma * 0.7 + inst * 0.3 : inst;
    fpsEl.textContent = `Render  ${fmtFps(renderEma)}`;
    if (renderEma < 20 && !lowWarned) {
      console.warn(`[virtual-arena] Render FPS below 20 (${renderEma.toFixed(1)}).`);
      lowWarned = true;
    } else if (renderEma >= 25) {
      lowWarned = false;
    }
  }
  lastRenderMs = now;
});

onInfer(() => {
  const now = performance.now();
  if (lastInferMs > 0) {
    const inst = 1000 / Math.max(now - lastInferMs, 1);
    inferEma = inferEma ? inferEma * 0.7 + inst * 0.3 : inst;
    infEl.textContent = `Infer   ${fmtFps(inferEma)}`;
  }
  lastInferMs = now;
});

// Inference rate limiter. Gates wantSeg/wantHands; cam-frame posting happens
// every rAF tick regardless (at display rate, throttled by worker busy gate).
let lastInferenceMs = -Infinity;

// Alternate seg + hands when foam is on (halves GPU work per inference tick).
let nextIsSeg = true;

let canvasRect = canvas.getBoundingClientRect();
window.addEventListener('resize', () => { canvasRect = canvas.getBoundingClientRect(); });

function stop() {
  running = false;
  cancelAnimationFrame(rafId);
  teardownSource();
  clearLandmarks();
  hideFoamFrom(0);
  renderEma = 0; inferEma = 0;
  lastRenderMs = -1; lastInferMs = -1;
  infEl.textContent = 'Infer     —';
  fpsEl.textContent = 'Render    —';
  toggleBtn.textContent = 'Start';
  toggleBtn.classList.remove('stop');
}

async function start() {
  clearError();
  toggleBtn.disabled = true;
  toggleBtn.classList.remove('pulse');
  try {
    await startSource();
    running = true;
    toggleBtn.textContent = 'Stop';
    toggleBtn.classList.add('stop');
    loop();
  } catch (e) {
    showError(e.message || 'Source start failed');
  } finally {
    toggleBtn.disabled = false;
  }
}

toggleBtn.addEventListener('click', () => running ? stop() : start());

function loop() {
  if (!running) return;
  rafId = requestAnimationFrame(loop);

  const now = performance.now();

  if (video.readyState < 2) return;

  // Shared source scale + offset (drives foam DOM and the shader).
  const S  = +camSize.value;
  const px = +camX.value;
  const py = +camY.value;
  const camOx = (1 - S) * (1 + px) * 0.5;
  const camOy = (1 - S) * (1 + py) * 0.5;

  if (foamToggle.checked) {
    applyFoam(getLandmarks(), mirrorInput.checked, canvasRect, S, camOx, camOy);
  }

  // Decide whether this tick's frame also carries an inference request.
  // canInfer() gates on the worker's inference in-flight — render keeps
  // flowing at rAF rate while a slow ONNX pass is still running.
  const wantInfer = canInfer() && (now - lastInferenceMs) >= 1000 / +infRateInput.value;
  const foam = foamToggle.checked;
  const wantSeg   = wantInfer && (foam ?  nextIsSeg : true);
  const wantHands = wantInfer && (foam ? !nextIsSeg : false);

  const sent = postFrame(video, now, {
    feather:    +featherInput.value,
    mirror:     mirrorInput.checked ? 1.0 : 0.0,
    camScale:   S,
    camOffsetX: camOx,
    camOffsetY: camOy,
    trim:       +trimInput.value,
  }, wantSeg, wantHands);

  if (sent && wantInfer) {
    lastInferenceMs = now;
    if (foam) nextIsSeg = !nextIsSeg;
  }
}

// Kick off the preload flow. The OffscreenCanvas rides along so init() can
// hand it to the worker in the same message as the model buffers.
preload(offscreenCanvas).catch(e => {
  progressSub.textContent = 'Preload failed — see error below';
  progressLabel.textContent = '';
  showError('Preload failed: ' + e.message);
});
