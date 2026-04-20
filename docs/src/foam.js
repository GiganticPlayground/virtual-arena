// Foam-finger DOM overlays driven by hand landmarks. Pools <img> elements
// keyed by hand index; hides unused ones each tick.

import { foamLayer } from './dom.js';
import { FOAM_IMG_URL } from './config.js';

const pool = [];

function getFoamEl(i) {
  while (pool.length <= i) {
    const el = document.createElement('img');
    el.className = 'foam';
    el.src = FOAM_IMG_URL;
    el.alt = '';
    el.style.display = 'none';
    foamLayer.appendChild(el);
    pool.push(el);
  }
  return pool[i];
}

export function hideFoamFrom(i) {
  for (let k = i; k < pool.length; k++) pool[k].style.display = 'none';
}

// Apply landmark data to foam fingers. Must match the shader's cam placement:
// cam covers canvas-UV [camOffsetX, camOffsetX+camScale] horizontally and
// [camOffsetY, camOffsetY+camScale] vertically, where canvas-UV y=1 is the
// TOP of the canvas. Landmarks are in video-space UV (uy=0 is top of video).
export function applyFoam(hands, mirror, canvasRect, camScale, camOffsetX, camOffsetY) {
  for (let i = 0; i < hands.length; i++) {
    const h = hands[i];
    const wrist = h[0], midMcp = h[9], midTip = h[12];
    let ux = (wrist.x + midMcp.x) * 0.5;
    const uy = (wrist.y + midMcp.y) * 0.5;
    let dirX = midMcp.x - wrist.x;
    const dirY = midMcp.y - wrist.y;
    let lenX = midTip.x - wrist.x;
    const lenY = midTip.y - wrist.y;
    if (mirror) { ux = 1 - ux; dirX = -dirX; lenX = -lenX; }

    const cx = canvasRect.left + (camOffsetX + ux * camScale) * canvasRect.width;
    const cy = canvasRect.top  + (1 - camOffsetY - camScale * (1 - uy)) * canvasRect.height;
    const pxDirX = dirX * camScale * canvasRect.width;
    const pxDirY = dirY * camScale * canvasRect.height;
    const pxLenX = lenX * camScale * canvasRect.width;
    const pxLenY = lenY * camScale * canvasRect.height;

    const angle  = Math.atan2(pxDirX, -pxDirY); // 0 = pointing up
    const handPx = Math.hypot(pxLenX, pxLenY);
    // PNG is rendered at 120x180; transform-origin at 50% 92% (near the
    // grip). Scale so the foam finger is ~1.8x the user's hand length.
    const scale = Math.max(0.2, (handPx * 1.8) / 180);
    const ox = 60, oy = 180 * 0.92;

    const el = getFoamEl(i);
    el.style.display = 'block';
    el.style.transform =
      `translate(${cx - ox}px, ${cy - oy}px) rotate(${angle}rad) scale(${scale})`;
  }
  hideFoamFrom(hands.length);
}
