// Arena background loading + "Custom" upload. The GL-backed bgTex lives in
// the worker now; this module decodes source images on the main thread and
// forwards them via setBackground(). We keep a main-thread Image cache so
// re-selecting an arena doesn't re-decode.

import { ARENAS } from './config.js';
import { arenaSel, uploadInput, showError } from './dom.js';
import { setBackground } from './worker-client.js';

// Cache of decoded Image objects keyed by URL. Filled during preload;
// upload flow adds to it on demand.
export const arenaImages = new Map();

// A 1x1 canvas filled with `color`. Cheap; used for solid-color ARENAS
// entries that never touch the network.
function solidColorCanvas(color) {
  const c = document.createElement('canvas');
  c.width = 1; c.height = 1;
  const ctx = c.getContext('2d');
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, 1, 1);
  return c;
}

export async function loadBackground(url) {
  let img = arenaImages.get(url);
  if (!img) {
    const entry = ARENAS.find(a => a.url === url);
    if (entry?.color) {
      img = solidColorCanvas(entry.color);
    } else {
      img = new Image();
      img.src = url;
      await img.decode();
    }
    arenaImages.set(url, img);
  }
  await setBackground(img);
}

export function populateArenaSelect() {
  for (const [i, a] of ARENAS.entries()) {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = a.label;
    arenaSel.appendChild(opt);
  }
}

export function wireArenaSelect() {
  arenaSel.addEventListener('change', () => {
    loadBackground(ARENAS[arenaSel.value].url)
      .catch(e => showError('Background load failed: ' + e.message));
  });
}

// User-supplied background: becomes (or updates) a single "Custom" entry in
// the dropdown. The blob URL is kept alive for the session and revoked when
// a new file is picked.
export function wireBackgroundUpload() {
  let customBlobUrl = null;
  uploadInput.addEventListener('change', async (e) => {
    const file = e.target.files?.[0];
    e.target.value = '';  // allow re-selecting the same file
    if (!file) return;
    if (customBlobUrl) URL.revokeObjectURL(customBlobUrl);
    customBlobUrl = URL.createObjectURL(file);

    const label = 'Custom — ' + file.name;
    let idx = ARENAS.findIndex(a => a.custom);
    if (idx === -1) {
      ARENAS.push({ label, url: customBlobUrl, custom: true });
      idx = ARENAS.length - 1;
      const opt = document.createElement('option');
      opt.value = idx;
      opt.textContent = label;
      arenaSel.appendChild(opt);
    } else {
      ARENAS[idx].url = customBlobUrl;
      ARENAS[idx].label = label;
      arenaSel.options[idx].textContent = label;
    }
    arenaSel.value = idx;
    try { await loadBackground(customBlobUrl); }
    catch (err) { showError('Failed to load uploaded image: ' + err.message); }
  });
}
