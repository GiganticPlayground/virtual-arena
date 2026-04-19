// Download all assets up front with a progress bar, then hand the model
// buffers to the inference worker. We fetch the model files ourselves
// (streaming) so we can show real byte progress, then pass the buffers to
// MediaPipe via `modelAssetBuffer` — no second download.

import { ARENAS, BINARY_SEG_URL, MULTICLASS_SEG_URL, HAND_MODEL_URL, WASM_URL, RVM_MODEL_URL, U2NETP_MODEL_URL, SILUETA_MODEL_URL, MODNET_MODEL_URL } from './config.js';
import { arenaImages, loadBackground } from './backgrounds.js';
import { initWorker } from './worker-client.js';
import {
  progressBar, progressLabel, progressSub, progressCta,
  preloadEl, toggleBtn,
} from './dom.js';

export async function preload(offscreenCanvas) {
  progressSub.textContent = 'Downloading models and backgrounds…';

  const items = [
    { key: 'segBinary',     url: BINARY_SEG_URL     },
    { key: 'segMulticlass', url: MULTICLASS_SEG_URL },
    { key: 'hand',          url: HAND_MODEL_URL     },
    // Solid-color arenas don't need fetching; loadBackground() synthesizes
    // a 1x1 canvas for them when selected.
    ...ARENAS.flatMap((a, i) => a.color ? [] : [{ key: 'arena', arenaIdx: i, url: a.url }]),
  ];

  // Open all requests in parallel, read Content-Length for each.
  await Promise.all(items.map(async (it) => {
    const res = await fetch(it.url);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${it.url}`);
    it.response = res;
    it.total = +res.headers.get('Content-Length') || 0;
    it.received = 0;
  }));
  const grandTotal = items.reduce((s, i) => s + i.total, 0);

  function updateProgress() {
    const received = items.reduce((s, i) => s + i.received, 0);
    const pct = grandTotal ? received / grandTotal : 0;
    progressBar.style.width = (pct * 100).toFixed(1) + '%';
    progressLabel.textContent =
      `${(received/1048576).toFixed(1)} / ${(grandTotal/1048576).toFixed(1)} MB`;
  }
  updateProgress();

  // Stream bodies, updating the bar as bytes arrive.
  await Promise.all(items.map(async (it) => {
    const reader = it.response.body.getReader();
    const chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      it.received += value.length;
      updateProgress();
    }
    const buf = new Uint8Array(it.received);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }
    it.buffer = buf;
  }));

  // Hand model buffers + the OffscreenCanvas off to the worker (all
  // transferred). After this call the main-thread Uint8Arrays are detached
  // and the <canvas> can no longer be drawn to from here.
  progressSub.textContent = 'Starting worker (compositor + inference)…';
  const wasmBaseUrl = new URL(WASM_URL, location.href).href;
  const binarySegBuffer     = items[0].buffer;
  const multiclassSegBuffer = items[1].buffer;
  const handBuffer          = items[2].buffer;
  await initWorker({
    canvas: offscreenCanvas,
    binarySegBuffer, multiclassSegBuffer, handBuffer,
    wasmBaseUrl,
    // Absolute URLs so the worker's fetch resolves against the page origin,
    // not the worker's module base.
    onnxModelUrls: {
      rvm:     new URL(RVM_MODEL_URL,     location.href).href,
      u2netp:  new URL(U2NETP_MODEL_URL,  location.href).href,
      silueta: new URL(SILUETA_MODEL_URL, location.href).href,
      modnet:  new URL(MODNET_MODEL_URL,  location.href).href,
    },
  });
  items[0].buffer = null;
  items[1].buffer = null;
  items[2].buffer = null;

  progressSub.textContent = 'Decoding backgrounds…';
  for (const it of items) {
    if (it.key !== 'arena') continue;
    const url = URL.createObjectURL(new Blob([it.buffer]));
    const img = new Image();
    img.src = url;
    await img.decode();
    arenaImages.set(ARENAS[it.arenaIdx].url, img);
  }
  await loadBackground(ARENAS[0].url);

  progressSub.textContent = 'Ready';
  progressCta.textContent = 'Click Start ↓';
  toggleBtn.disabled = false;
  toggleBtn.classList.add('pulse');
  preloadEl.classList.add('done');
  setTimeout(() => { preloadEl.style.display = 'none'; }, 500);
}
