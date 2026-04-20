// Single source of DOM element handles. Every other module pulls what it
// needs from here so IDs are declared in one place.

const $ = (id) => document.getElementById(id);

export const canvas           = $('gl');
export const video            = $('video');
export const fpsEl            = $('fps');
export const infEl            = $('inf');
export const statusEl         = $('status');
export const toggleBtn        = $('toggle');
export const arenaSel         = $('arena');
export const featherInput     = $('feather');
export const featherVal       = $('featherVal');
export const mirrorInput      = $('mirror');
export const uploadInput      = $('upload');
export const uploadVideoInput = $('uploadVideo');
export const foamToggle       = $('foam');
export const foamLayer        = $('foam-layer');
export const handCountSel     = $('handCount');
export const infRateInput     = $('infRate');
export const infRateVal       = $('infRateVal');
export const sourceSel        = $('source');
export const camSize          = $('camSize');
export const camSizeVal       = $('camSizeVal');
export const camX             = $('camX');
export const camXVal          = $('camXVal');
export const camY             = $('camY');
export const camYVal          = $('camYVal');
export const modelSel         = $('model');
export const rvmQualitySel    = $('rvmQuality');
export const trimInput        = $('trim');
export const trimVal          = $('trimVal');
export const preloadEl        = $('preload');
export const progressBar      = $('progressBar');
export const progressLabel    = $('progressLabel');
export const progressSub      = $('progressSub');
export const progressCta      = $('progressCta');

export function showError(msg) {
  console.error(msg);
  statusEl.textContent = msg;
  statusEl.style.display = 'block';
}

export function clearError() {
  statusEl.style.display = 'none';
}
