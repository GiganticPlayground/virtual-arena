// Shared constants. Paths are all relative to the served root (docs/).

// Backgrounds. Drop new .jpg/.png in docs/arenas/ and add here.
export const ARENAS = [
  { label: 'Background 1', url: './arenas/1.png' },
  { label: 'Background 2', url: './arenas/2.png' },
  { label: 'Background 3', url: './arenas/3.jpg' },
];

// Multiclass selfie segmenter (category 0 = background; 1-5 = person parts).
// Hand landmarker is used for the foam-finger overlay.
export const MODEL_URL      = './models/selfie_multiclass_256x256.tflite';
export const HAND_MODEL_URL = './models/hand_landmarker.task';
export const WASM_URL       = './vision/wasm';

// Foam finger art: inline SVG data URL (red mitt with raised index finger
// and "#1" on the palm). Drawn pointing up; rotated at runtime to match
// the hand orientation.
export const FOAM_SVG_URL = 'data:image/svg+xml;utf8,' + encodeURIComponent(
`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 180">
  <g stroke="#5a0a0a" stroke-width="3" stroke-linejoin="round">
    <ellipse cx="18" cy="118" rx="14" ry="22" fill="#d84027"/>
    <rect x="10" y="92" width="80" height="78" rx="22" fill="#d84027"/>
    <rect x="39" y="4" width="22" height="100" rx="11" fill="#d84027"/>
  </g>
  <text x="52" y="150" font-family="Impact, 'Arial Black', sans-serif"
        font-weight="900" font-size="44" fill="#fff7e0"
        stroke="#5a0a0a" stroke-width="1.5" text-anchor="middle">#1</text>
</svg>`);
