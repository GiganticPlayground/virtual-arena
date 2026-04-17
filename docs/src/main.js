// App entry point. Wires up UI controls, owns the Start/Stop lifecycle and
// the rAF loop. Everything stateful lives in its own module; this file is
// glue that coordinates them.

import {
  canvas, video, fpsEl, infEl, toggleBtn,
  featherInput, featherVal, mirrorInput, foamToggle,
  infRateInput, infRateVal, sourceSel, uploadVideoInput,
  camSize, camSizeVal, camX, camXVal, camY, camYVal,
  progressSub, progressLabel,
  showError, clearError,
} from './dom.js';
import { uploadCamFrame, drawFrame } from './renderer.js';
import { tryInfer, getLandmarks, clearLandmarks, onResult } from './worker-client.js';
import { startSource, teardownSource } from './source.js';
import { applyFoam, hideFoamFrom } from './foam.js';
import {
  populateArenaSelect, wireArenaSelect, wireBackgroundUpload, getBgAspect,
} from './backgrounds.js';
import { preload } from './preload.js';

// ---- Init UI ----------------------------------------------------------
populateArenaSelect();
wireArenaSelect();
wireBackgroundUpload();

// Live-value labels next to sliders.
for (const [input, span] of [[camSize, camSizeVal], [camX, camXVal], [camY, camYVal]]) {
  const update = () => { span.textContent = (+input.value).toFixed(2); };
  input.addEventListener('input', update);
  update();
}
infRateInput.addEventListener('input', () => { infRateVal.textContent = infRateInput.value; });
featherInput.addEventListener('input', () => { featherVal.textContent = (+featherInput.value).toFixed(1); });
foamToggle.addEventListener('change', () => {
  if (!foamToggle.checked) hideFoamFrom(0);
});

// Live source swap while running.
sourceSel.addEventListener('change', async () => {
  if (!running) return;
  toggleBtn.disabled = true;
  try { await startSource(); }
  catch (e) { showError(e.message); stop(); }
  finally { toggleBtn.disabled = false; }
});

// Uploaded video: single "Custom" entry in the Source dropdown. Blob URL is
// kept alive for the session and revoked when a new file is picked.
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

// Render FPS accumulators.
let lastFrame = performance.now();
let fpsAcc = 0, fpsFrames = 0;
let lowWarned = false;

// Inference FPS accumulators. Bumped by the onResult callback; drained by
// the render loop on the same 15-frame cadence as the render HUD.
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

// Inference rate limiter. The compositor always draws every rAF tick; only
// the dispatch-frame-to-worker step is gated by the user's Hz slider.
let lastInferenceMs = -Infinity;

// Cached canvas bounding rect for landmark → viewport mapping.
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
      console.warn(`[virtual-arena] FPS dropped below 20 (${fps.toFixed(1)}). Consider a smaller model or lower capture resolution.`);
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
      // No inference results for ~2 windows → reset.
      infEl.textContent = 'Infer     —';
      lastInfMs = -1;
    }
  }

  if (video.readyState < 2) return;

  uploadCamFrame(video);

  // Rate-limit inference dispatch. tryInfer() returns false if the worker
  // is busy / not ready / broken; in that case we leave lastInferenceMs
  // alone so the next tick can try again immediately.
  if ((now - lastInferenceMs) >= 1000 / +infRateInput.value) {
    if (tryInfer(video, now, foamToggle.checked)) {
      lastInferenceMs = now;
    }
  }

  // Shared scale + offset for shader and foam-finger DOM placement.
  const S  = +camSize.value;
  const px = +camX.value;
  const py = +camY.value;
  const camOx = (1 - S) * (1 + px) * 0.5;
  const camOy = (1 - S) * (1 + py) * 0.5;

  if (foamToggle.checked) {
    applyFoam(getLandmarks(), mirrorInput.checked, canvasRect, S, camOx, camOy);
  }

  drawFrame({
    bgAspect:   getBgAspect(),
    feather:    +featherInput.value,
    mirror:     mirrorInput.checked,
    camScale:   S,
    camOffsetX: camOx,
    camOffsetY: camOy,
  });
}

// Kick off the preload flow.
preload().catch(e => {
  progressSub.textContent = 'Preload failed — see error below';
  progressLabel.textContent = '';
  showError('Preload failed: ' + e.message);
});
