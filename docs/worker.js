// Inference worker. Owns both MediaPipe tasks and runs inference on
// VideoFrame/ImageBitmap instances posted from the main thread.

// MediaPipe's loader tries `importScripts(url)` first (classic worker path)
// and on TypeError falls back to `await self.import(url)` — which it
// expects US to define. The URL points to a UMD script (vision_wasm_internal.js),
// not an ES module, so dynamic `import()` won't work. We fetch + indirect-eval
// so the script's top-level `var ModuleFactory` becomes a global on `self`,
// exactly like `importScripts` would do in a classic worker.
self.import = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`self.import fetch failed: ${res.status} ${url}`);
  const src = await res.text();
  (0, eval)(src); // indirect eval → global scope
};

import { FilesetResolver, ImageSegmenter, HandLandmarker } from './vision/vision_bundle.mjs';

let segmenter      = null;
let handLandmarker = null;

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      const vision = await FilesetResolver.forVisionTasks(msg.wasmBaseUrl);
      segmenter = await ImageSegmenter.createFromOptions(vision, {
        baseOptions: { modelAssetBuffer: msg.segBuffer, delegate: 'GPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetBuffer: msg.handBuffer, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 4,
      });
      self.postMessage({ type: 'ready' });
      return;
    }

    if (msg.type === 'infer') {
      let mask = null, maskW = 0, maskH = 0;
      segmenter.segmentForVideo(msg.frame, msg.ts, (r) => {
        const cm = r.categoryMask;
        if (cm) {
          maskW = cm.width;
          maskH = cm.height;
          mask  = cm.getAsUint8Array();
          cm.close();
        }
        r.close?.();
      });

      let landmarks = [];
      if (msg.wantHands) {
        const hr = handLandmarker.detectForVideo(msg.frame, msg.ts);
        landmarks = hr?.landmarks ?? [];
      }

      // Critical: VideoFrame must be closed or Chrome back-pressures the decoder.
      msg.frame.close?.();

      self.postMessage(
        { type: 'result', ts: msg.ts, mask, maskW, maskH, landmarks },
        mask ? [mask.buffer] : []
      );
      return;
    }
  } catch (err) {
    // If an infer failed, close the frame to avoid leaking it.
    if (msg?.type === 'infer') msg.frame?.close?.();
    self.postMessage({
      type: 'error',
      stage: msg?.type === 'init' ? 'init' : 'infer',
      message: err?.message || String(err),
      stack:   err?.stack,
    });
  }
};

self.onerror = (e) => {
  self.postMessage({
    type: 'error',
    stage: 'runtime',
    message: e?.message || String(e),
  });
};
