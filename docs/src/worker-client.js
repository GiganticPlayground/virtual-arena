// Wraps ./worker.js. Owns the ready/busy/broken state and the one-in-flight
// gate. The main-thread render loop calls tryInfer(); results flow straight
// into the renderer's mask texture and a landmark cache.

import { showError } from './dom.js';
import { uploadMask } from './renderer.js';

const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });

let ready  = false;
let busy   = false;   // one-in-flight gate
let broken = false;   // latched on init/runtime errors
let lastLandmarks = [];

let readyResolve;
const readyPromise = new Promise(r => { readyResolve = r; });

// Optional callback fired whenever a 'result' reply arrives. main.js uses
// this for the inference-rate HUD.
let onResultCb = null;
export function onResult(cb) { onResultCb = cb; }

worker.addEventListener('message', (e) => {
  const m = e.data;
  if (m.type === 'ready') {
    ready = true;
    readyResolve();
  } else if (m.type === 'result') {
    busy = false;
    if (m.mask) uploadMask(m.mask, m.maskW, m.maskH);
    // null => the worker didn't run the hand landmarker this tick (staggered);
    // keep whatever we had. [] is a real "zero hands detected" result.
    if (m.landmarks != null) lastLandmarks = m.landmarks;
    onResultCb?.();
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

// Hand both model buffers + wasm base URL to the worker. Transferring
// detaches the passed Uint8Arrays — callers should null their references.
export function initWorker({ segBuffer, handBuffer, wasmBaseUrl }) {
  worker.postMessage(
    { type: 'init', segBuffer, handBuffer, wasmBaseUrl },
    [segBuffer.buffer, handBuffer.buffer],
  );
  return readyPromise;
}

// Attempt to dispatch a frame. Returns true iff a frame was actually sent;
// caller uses that to gate its rate limiter. wantSeg and wantHands are
// independent so the caller can stagger them across successive inferences.
export function tryInfer(videoEl, ts, wantSeg, wantHands) {
  if (!ready || busy || broken) return false;
  if (!wantSeg && !wantHands) return false;
  busy = true;
  try {
    if (typeof VideoFrame !== 'undefined') {
      const frame = new VideoFrame(videoEl);  // zero-copy on Chrome
      worker.postMessage({ type: 'infer', frame, ts, wantSeg, wantHands }, [frame]);
    } else {
      // Fallback: async ImageBitmap. Fire-and-forget so the rAF loop stays sync.
      createImageBitmap(videoEl).then(bmp => {
        worker.postMessage({ type: 'infer', frame: bmp, ts, wantSeg, wantHands }, [bmp]);
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

export function getLandmarks()   { return lastLandmarks; }
export function clearLandmarks() { lastLandmarks = []; }
