// Inference + compositor worker. Owns:
//   - MediaPipe segmenter + hand landmarker
//   - OffscreenCanvas WebGL2 context + shader program
//   - bg/cam/mask textures + all draw-time state
// Main thread does only: DOM, <video>, foam-finger overlay, rAF + capture.

// MediaPipe's loader tries `importScripts(url)` first (classic worker path)
// and on TypeError falls back to `await self.import(url)` — which it
// expects US to define. The URL points to a UMD script, not an ES module,
// so dynamic `import()` won't work. Fetch + indirect-eval puts the script's
// top-level `var`s on `self` like `importScripts` would in a classic worker.
self.import = async (url) => {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`self.import fetch failed: ${res.status} ${url}`);
  const src = await res.text();
  (0, eval)(src); // indirect eval → global scope
};

import { FilesetResolver, ImageSegmenter, HandLandmarker } from '../vision/vision_bundle.mjs';
import { VS, FS } from './shaders.js';

// ---- GL state (populated in initGL) -----------------------------------
let canvas = null;
let gl = null;
let u = null;       // uniform locations keyed by uniform name
let vao = null;
let bgTex = null, camTex = null, maskTex = null;
let bgAspect = 16 / 9;

// Latest shader parameters pushed from main. Every frame message can
// carry an update.
let params = {
  feather:    1.5,
  mirror:     1.0,
  camScale:   1.0,
  camOffsetX: 0,
  camOffsetY: 0,
  trim:       0.0,
};

// ---- MediaPipe --------------------------------------------------------
// Both segmenters stay loaded; setModel swaps the active reference at
// runtime with no re-init cost.
let segmenterBinary     = null;
let segmenterMulticlass = null;
let activeSegmenter     = null;
let handLandmarker      = null;

// ---- GL helpers -------------------------------------------------------
function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error('Shader compile: ' + gl.getShaderInfoLog(s));
  }
  return s;
}

function makeTex() {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return t;
}

function initGL(offscreen) {
  canvas = offscreen;
  gl = canvas.getContext('webgl2', {
    alpha: false, antialias: false, premultipliedAlpha: false,
  });
  if (!gl) throw new Error('WebGL2 unavailable in worker context.');

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error('Program link: ' + gl.getProgramInfoLog(prog));
  }
  gl.useProgram(prog);

  vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1, -1,  1,
    -1,  1,  1, -1,  1,  1,
  ]), gl.STATIC_DRAW);
  const aPos = gl.getAttribLocation(prog, 'a_pos');
  gl.enableVertexAttribArray(aPos);
  gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

  u = Object.fromEntries([
    'u_bg', 'u_cam', 'u_mask',
    'u_feather', 'u_maskSize', 'u_mirror',
    'u_bgScale', 'u_bgOffset',
    'u_camScale', 'u_camOffset',
    'u_trim', 'u_maskInvert',
  ].map(n => [n, gl.getUniformLocation(prog, n)]));

  bgTex   = makeTex();
  camTex  = makeTex();
  maskTex = makeTex();
}

function uploadCam(frame) {
  gl.bindTexture(gl.TEXTURE_2D, camTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, frame);
}

function uploadMask(data, w, h) {
  gl.bindTexture(gl.TEXTURE_2D, maskTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, data);
  gl.uniform2f(u.u_maskSize, w, h);
}

function draw() {
  const canvasAspect = canvas.width / canvas.height;
  let sx = 1, sy = 1, ox = 0, oy = 0;
  if (bgAspect > canvasAspect) { sx = canvasAspect / bgAspect; ox = (1 - sx) * 0.5; }
  else                         { sy = bgAspect / canvasAspect; oy = (1 - sy) * 0.5; }

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, bgTex);   gl.uniform1i(u.u_bg,   0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, camTex);  gl.uniform1i(u.u_cam,  1);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, maskTex); gl.uniform1i(u.u_mask, 2);
  gl.uniform1f(u.u_feather,   params.feather);
  gl.uniform1f(u.u_mirror,    params.mirror);
  gl.uniform2f(u.u_bgScale,   sx, sy);
  gl.uniform2f(u.u_bgOffset,  ox, oy);
  gl.uniform1f(u.u_camScale,  params.camScale);
  gl.uniform2f(u.u_camOffset, params.camOffsetX, params.camOffsetY);
  gl.uniform1f(u.u_trim,      params.trim);
  // Binary selfie_segmenter inverts foreground vs multiclass; flip the
  // shader's mask interpretation when it's the active model.
  gl.uniform1f(u.u_maskInvert, activeSegmenter === segmenterBinary ? 1.0 : 0.0);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

// ---- Message handler --------------------------------------------------
self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      initGL(msg.canvas);

      const vision = await FilesetResolver.forVisionTasks(msg.wasmBaseUrl);
      const segOpts = (buf) => ({
        baseOptions: { modelAssetBuffer: buf, delegate: 'GPU' },
        runningMode: 'VIDEO',
        outputCategoryMask: true,
        outputConfidenceMasks: false,
      });
      segmenterBinary     = await ImageSegmenter.createFromOptions(vision, segOpts(msg.binarySegBuffer));
      segmenterMulticlass = await ImageSegmenter.createFromOptions(vision, segOpts(msg.multiclassSegBuffer));
      activeSegmenter     = segmenterBinary;  // cleaner default
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetBuffer: msg.handBuffer, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 4,
      });
      self.postMessage({ type: 'ready' });
      return;
    }

    if (msg.type === 'setModel') {
      activeSegmenter = msg.model === 'multiclass' ? segmenterMulticlass : segmenterBinary;
      return;
    }

    if (msg.type === 'background') {
      const bm = msg.bitmap;
      // Bitmap arrives pre-flipped (imageOrientation:'flipY' on main),
      // so we upload without another flip to land it right-side-up.
      gl.bindTexture(gl.TEXTURE_2D, bgTex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, bm);
      bgAspect = bm.width / bm.height;
      bm.close?.();
      return;
    }

    if (msg.type === 'frame') {
      const frame = msg.frame;
      if (msg.params) params = { ...params, ...msg.params };

      uploadCam(frame);

      if (msg.wantSeg) {
        activeSegmenter.segmentForVideo(frame, msg.ts, (r) => {
          const cm = r.categoryMask;
          if (cm) {
            const data = cm.getAsUint8Array();
            uploadMask(data, cm.width, cm.height);
            cm.close();
          }
          r.close?.();
        });
      }

      let landmarks = null;
      if (msg.wantHands) {
        const hr = handLandmarker.detectForVideo(frame, msg.ts);
        landmarks = hr?.landmarks ?? [];
      }

      // Critical: VideoFrame must be closed or Chrome back-pressures the decoder.
      frame.close?.();

      draw();

      self.postMessage({
        type: 'rendered',
        ts: msg.ts,
        landmarks,
        hadSeg:   !!msg.wantSeg,
        hadHands: !!msg.wantHands,
      });
      return;
    }
  } catch (err) {
    if (msg?.type === 'frame') msg.frame?.close?.();
    self.postMessage({
      type: 'error',
      stage: msg?.type === 'init'  ? 'init'
           : msg?.type === 'frame' ? 'frame'
           : 'runtime',
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
