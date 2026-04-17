// Arena background loading + "Custom" upload. Owns the decoded-Image cache
// and the aspect ratio used by the renderer for cover-fit math.

import { ARENAS } from './config.js';
import { arenaSel, uploadInput, showError } from './dom.js';
import { uploadBackgroundImage } from './renderer.js';

// Cache of decoded Image objects keyed by URL. Filled during preload;
// upload flow adds to it on demand.
export const arenaImages = new Map();
let bgAspect = 16 / 9;

export function getBgAspect() { return bgAspect; }

export async function loadBackground(url) {
  let img = arenaImages.get(url);
  if (!img) {
    img = new Image();
    img.src = url;
    await img.decode();
    arenaImages.set(url, img);
  }
  bgAspect = uploadBackgroundImage(img);
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
