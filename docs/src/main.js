// App entry point. With the OffscreenCanvas refactor, the main thread does
// only: DOM control wiring, <video> lifecycle, foam-finger overlay transforms,
// and per-rAF cam frame capture + postMessage. All WebGL + MediaPipe work is
// in ./worker.js.

import {
  canvas, video, fpsEl, infEl, toggleBtn,
  featherInput, featherVal, mirrorInput, foamToggle,
  infRateInput, infRateVal, sourceSel, uploadVideoInput,
  camSize, camSizeVal, camX, camXVal, camY, camYVal,
  modelSel, trimInput, trimVal,
  progressSub, progressLabel,
  showError, clearError,
} from './dom.js';
import { postFrame, getLandmarks, clearLandmarks, onResult, setActiveModel } from './worker-client.js';
import { startSource, teardownSource } from './source.js';
import { applyFoam, hideFoamFrom } from './foam.js';
import { populateArenaSelect, wireArenaSelect, wireBackgroundUpload } from './backgrounds.js';
import { preload } from './preload.js';

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
modelSel.addEventListener('change',    () => { setActiveModel(modelSel.value); });
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

let lastFrame = performance.now();
let fpsAcc = 0, fpsFrames = 0;
let lowWarned = false;

// Inference FPS accumulators. Bumped by onResult (fires when a worker
// 'rendered' reply carries inference results).
let lastInfMs = -1;
let infAcc = 0, infFrames = 0;
let infStaleTimer = 0;
onResult(() => {
  const inow = performance.now();
  if (lastInfMs > 0) {
    infAcc += 1000 / Math.max(inow - lastInfMs, 1);
    infFrames++;
  }
  lastInfMs = inow;
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
  lastInfMs = -1; infAcc = 0; infFrames = 0; infStaleTimer = 0;
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
    lastFrame = performance.now();
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
  const dt  = now - lastFrame;
  lastFrame = now;
  fpsAcc += 1000 / Math.max(dt, 1);
  fpsFrames++;
  if (fpsFrames >= 15) {
    const fps = fpsAcc / fpsFrames;
    fpsAcc = 0; fpsFrames = 0;
    fpsEl.textContent = `Render  ${fps.toFixed(0).padStart(3, ' ')}`;
    if (fps < 20 && !lowWarned) {
      console.warn(`[virtual-arena] FPS dropped below 20 (${fps.toFixed(1)}).`);
      lowWarned = true;
    } else if (fps >= 25) {
      lowWarned = false;
    }
    if (infFrames > 0) {
      const inf = infAcc / infFrames;
      infAcc = 0; infFrames = 0;
      infEl.textContent = `Infer   ${inf.toFixed(0).padStart(3, ' ')}`;
      infStaleTimer = 0;
    } else if (++infStaleTimer >= 2) {
      infEl.textContent = 'Infer     —';
      lastInfMs = -1;
    }
  }

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
  const wantInfer = (now - lastInferenceMs) >= 1000 / +infRateInput.value;
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
