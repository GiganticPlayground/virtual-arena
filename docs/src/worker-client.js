// Wraps ./worker.js. The worker owns both inference AND the OffscreenCanvas
// compositor now — main posts frames + params and gets back landmarks only.

import { showError } from './dom.js';

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

let ready  = false;
let busy   = false;   // one-in-flight gate
let broken = false;   // latched on init/runtime errors
let lastLandmarks = [];

let readyResolve;
const readyPromise = new Promise(r => { readyResolve = r; });

// Fires whenever a 'rendered' reply carries inference results. main.js
// uses this to bump the Infer HUD.
let onResultCb = null;
export function onResult(cb) { onResultCb = cb; }

worker.addEventListener('message', (e) => {
  const m = e.data;
  if (m.type === 'ready') {
    ready = true;
    readyResolve();
  } else if (m.type === 'rendered') {
    busy = false;
    // null => worker didn't run hand landmarker this tick (staggered);
    // keep the cached value. [] is a real "zero hands detected" result.
    if (m.landmarks != null) lastLandmarks = m.landmarks;
    if (m.hadSeg || m.hadHands) onResultCb?.();
  } else if (m.type === 'error') {
    busy = false;
    if (m.stage === 'init') broken = true;
    showError(`Worker ${m.stage}: ${m.message}`);
    console.error('[worker]', m.stage, m.message, m.stack);
  }
});
worker.addEventListener('error', (e) => {
  broken = true;
  showError('Worker crashed: ' + (e.message || 'unknown'));
});

// Transfer the OffscreenCanvas + all model buffers in one init message.
// All four are detached from main after this call.
export function initWorker({ canvas, binarySegBuffer, multiclassSegBuffer, handBuffer, wasmBaseUrl }) {
  worker.postMessage(
    { type: 'init', canvas, binarySegBuffer, multiclassSegBuffer, handBuffer, wasmBaseUrl },
    [canvas, binarySegBuffer.buffer, multiclassSegBuffer.buffer, handBuffer.buffer],
  );
  return readyPromise;
}

// Swap which segmenter runs on subsequent frames. 'binary' | 'multiclass'.
export function setActiveModel(which) {
  worker.postMessage({ type: 'setModel', model: which });
}

// Post a cam frame + current render params (+ optional inference flags) to
// the worker. Returns true iff a frame was actually sent. One-in-flight:
// caller's rAF will skip while the worker is busy.
export function postFrame(videoEl, ts, params, wantSeg, wantHands) {
  if (!ready || busy || broken) return false;
  busy = true;
  try {
    if (typeof VideoFrame !== 'undefined') {
      const frame = new VideoFrame(videoEl);  // zero-copy on Chrome
      worker.postMessage(
        { type: 'frame', frame, ts, params, wantSeg, wantHands },
        [frame],
      );
    } else {
      // Fallback: async ImageBitmap. Fire-and-forget so the rAF loop stays sync.
      createImageBitmap(videoEl).then(bmp => {
        worker.postMessage(
          { type: 'frame', frame: bmp, ts, params, wantSeg, wantHands },
          [bmp],
        );
      }).catch(err => {
        busy = false;
        console.warn('Frame capture failed', err);
      });
    }
    return true;
  } catch (err) {
    busy = false;
    console.warn('Frame capture failed', err);
    return false;
  }
}

// Upload a background to the worker's bgTex. `source` is anything
// createImageBitmap accepts (HTMLImageElement, Blob, ImageBitmap).
// imageOrientation:'flipY' asks the browser to flip rows during decode so
// the bitmap lands in WebGL bottom-up order; the worker uploads it with
// UNPACK_FLIP_Y_WEBGL=false to match. Without this, Chrome's ImageBitmap
// path leaves the image top-down and FLIP_Y is effectively ignored,
// producing an upside-down background.
export async function setBackground(source) {
  const bitmap = await createImageBitmap(source, { imageOrientation: 'flipY' });
  worker.postMessage({ type: 'background', bitmap }, [bitmap]);
}

export function getLandmarks()   { return lastLandmarks; }
export function clearLandmarks() { lastLandmarks = []; }
