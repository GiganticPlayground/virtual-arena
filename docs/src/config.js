// Shared constants. Paths are all relative to the served root (docs/).

// Backgrounds. Drop new .jpg/.png in docs/arenas/ and add here.
// For a solid color, set `color` (any CSS color string) and leave url as a
// unique sentinel — preload skips fetching, and loadBackground synthesizes
// a 1x1 canvas.
export const ARENAS = [
  { label: 'Background 1', url: './arenas/1.png' },
  { label: 'Background 2', url: './arenas/2.png' },
  { label: 'Background 3', url: './arenas/3.jpg' },
  { label: 'Solid White',  url: 'solid:white', color: '#ffffff' },
];

// Two segmenter models the user can switch between at runtime:
//   - Binary (selfie_segmenter): single person-vs-background decision;
//     much cleaner silhouette in practice (chairs, props stay out).
//   - Multiclass: per-pixel person-part classification (0 = bg; 1-5 =
//     hair/body/face/clothes/others). Tends to drag in objects touching
//     the subject, so it's here as an opt-in.
// Hand landmarker is used for the foam-finger overlay.
export const BINARY_SEG_URL     = './models/selfie_segmenter.tflite';
export const MULTICLASS_SEG_URL = './models/selfie_multiclass_256x256.tflite';
export const HAND_MODEL_URL     = './models/hand_landmarker.task';
export const WASM_URL           = './vision/wasm';

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
