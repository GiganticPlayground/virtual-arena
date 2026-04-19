// Wraps ./worker.js. The worker owns both inference AND the OffscreenCanvas
// compositor now — main posts frames + params and gets back landmarks only.

import { showError } from './dom.js';

// Cache-bust the worker script with a load-time query string. Workers are
// cached aggressively by the browser and normal reloads can re-run the
// previous script. The query is ignored by the server but changes the URL
// identity so the Worker constructor always fetches fresh.
const workerUrl = new URL('./worker.js', import.meta.url);
workerUrl.searchParams.set('v', String(Date.now()));
const worker = new Worker(workerUrl, { type: 'module' });

let ready  = false;
let busy   = false;   // one-in-flight gate
let broken = false;   // latched on init/runtime errors
let lastLandmarks = [];

let readyResolve;
const readyPromise = new Promise(r => { readyResolve = r; });

// Fires on every 'rendered' reply (one per worker draw). main.js uses
// this for the actual canvas-update FPS.
let onRenderCb = null;
export function onRender(cb) { onRenderCb = cb; }

// Fires only when the 'rendered' reply carried seg or hands. main.js
// uses this for the Infer FPS HUD.
let onInferCb = null;
export function onInfer(cb) { onInferCb = cb; }

// Fires when the worker reports RVM load state transitions
// ('loading' | 'ready' | 'error').
let onRvmStatusCb = null;
export function onRvmStatus(cb) { onRvmStatusCb = cb; }

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
    onRenderCb?.();
    if (m.hadSeg || m.hadHands) onInferCb?.();
  } else if (m.type === 'rvmStatus') {
    onRvmStatusCb?.(m);
    if (m.status === 'error') showError(`RVM: ${m.message}`);
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

// Transfer the OffscreenCanvas + all MediaPipe model buffers in one init
// message. The ONNX models (RVM, u2netp, silueta) aren't preloaded — their
// URLs are passed along so the worker can fetch them lazily the first time
// the user selects one.
export function initWorker({ canvas, binarySegBuffer, multiclassSegBuffer, handBuffer, wasmBaseUrl, onnxModelUrls }) {
  worker.postMessage(
    { type: 'init', canvas, binarySegBuffer, multiclassSegBuffer, handBuffer, wasmBaseUrl, onnxModelUrls },
    [canvas, binarySegBuffer.buffer, multiclassSegBuffer.buffer, handBuffer.buffer],
  );
  return readyPromise;
}

// Swap which segmenter runs on subsequent frames. 'binary' | 'multiclass'.
export function setActiveModel(which) {
  worker.postMessage({ type: 'setModel', model: which });
}

// Set RVM input-resolution preset. 'quality' | 'balanced' | 'fast'.
// Safe before RVM is loaded — worker persists the setting.
export function setRvmQuality(quality) {
  worker.postMessage({ type: 'setRvmQuality', quality });
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
