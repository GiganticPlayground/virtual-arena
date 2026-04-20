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
let activeSegmenter     = null;  // one of the MediaPipe tasks above, or null when RVM is active
let handLandmarker      = null;

// ---- ONNX-based segmenters (via onnxruntime-web) ----------------------
// Each ONNX model is lazy-loaded on first selection. RVM has its own
// recurrent-state path; u2netp and silueta share a generic per-frame path.
let activeModel = 'binary';   // 'binary' | 'multiclass' | 'rvm' | 'u2netp' | 'silueta' | 'modnet'
let ortModule   = null;
let onnxModelUrls = {};       // { rvm, u2netp, silueta } → absolute URLs
let onnxSessions  = {};       // { rvm: session, u2netp: session, ... }
let onnxLoading   = {};       // per-model loading flags
let rvmState      = null;     // { r1i, r2i, r3i, r4i } tensors (RVM only)
let onnxCanvas    = null;     // shared OffscreenCanvas; resized per-model
let onnxCtx       = null;

// Per-model preprocessing. "simple" models: resize to WxH, normalize with
// ImageNet mean/std, run, minmax-normalize the output, upload as mask.
// `eps` is the ORT executionProviders list — WebGPU works for these
// feed-forward models (no recurrent state to get numerically corrupted
// like RVM had), so we prefer it for the 10x speedup and fall back to
// WASM if WebGPU can't handle some op in the graph.
// Default dims in each entry are overridden by the active ONNX_QUALITIES
// preset at runtime.
// `postProcess` selects how the raw output maps to [0,1] alpha:
//   'minmax'  — rembg-style per-frame rescale; good for saliency nets whose
//               output range drifts (u2netp, silueta).
//   'sigmoid' — element-wise 1/(1+e^-x); correct for graphs whose last op is
//               a raw logit (BiRefNet). Using minmax here would amplify noise
//               on frames with no strong foreground.
//   'scale'   — direct v*255; for models whose output is already in [0,1]
//               (modnet). Skips the two-pass min/max scan.
const SIMPLE_ONNX_MODELS = {
  u2netp:  { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], eps: ['webgpu', 'wasm'], postProcess: 'minmax' },
  silueta: { mean: [0.485, 0.456, 0.406], std: [0.229, 0.224, 0.225], eps: ['webgpu', 'wasm'], postProcess: 'minmax' },
  // MODNet was trained with normalization to [-1, 1] (mean=0.5, std=0.5),
  // not ImageNet. Its output is already in [0, 1], so a direct scale is
  // correct and cheaper than the two-pass minmax.
  modnet:  { mean: [0.5, 0.5, 0.5],       std: [0.5, 0.5, 0.5],       eps: ['webgpu', 'wasm'], postProcess: 'scale' },
};

// Input-resolution presets. Only applies to RVM, which was exported with
// dynamic H/W so it can accept any size. u2netp and silueta were exported
// with static [1,3,320,320] inputs — the ONNX session rejects anything
// else, so their dims are fixed regardless of the user's preset choice.
const ONNX_QUALITIES = {
  rvm: { quality: { w: 512, h: 288 }, balanced: { w: 384, h: 224 }, fast: { w: 256, h: 160 } },
};
const ONNX_FIXED_DIMS = {
  u2netp:  { w: 320, h: 320 },
  silueta: { w: 320, h: 320 },
  modnet:  { w: 512, h: 512 },
};
let onnxQuality = 'quality';
function onnxDims(model) {
  if (ONNX_FIXED_DIMS[model]) return ONNX_FIXED_DIMS[model];
  const p = ONNX_QUALITIES[model];
  return (p && p[onnxQuality]) || { w: 320, h: 320 };
}
// Bumped whenever quality changes. rvmInfer captures it at the start of a
// run and throws away the returned recurrent state if the generation has
// advanced during its await — otherwise old-shape tensors would overwrite
// the freshly-reset placeholders and the next run would shape-mismatch.
let rvmQualityGen = 0;

function ensureOnnxCanvas(w, h) {
  if (!onnxCanvas) {
    onnxCanvas = new OffscreenCanvas(w, h);
    onnxCtx = onnxCanvas.getContext('2d', { willReadFrequently: true });
  } else if (onnxCanvas.width !== w || onnxCanvas.height !== h) {
    onnxCanvas.width = w;
    onnxCanvas.height = h;
  }
}

// Scratch buffers shared across consecutive simpleOnnxInfer/rvmInfer calls.
// Reused rather than reallocated to cut GC pressure in the hot path. The
// inference gate on main (inferBusy) guarantees no concurrent access, so
// a single buffer per size is safe.
let onnxSrcBuf     = null;  // Float32Array(n*3) input plane
let onnxSrcSize    = 0;      // n = W*H of the last alloc
let onnxOutBytes   = null;  // Uint8Array(n) mask bytes
let onnxOutSize    = 0;
function ensureSrcBuf(n) {
  if (onnxSrcSize !== n) { onnxSrcBuf = new Float32Array(n * 3); onnxSrcSize = n; }
  return onnxSrcBuf;
}
function ensureOutBuf(n) {
  if (onnxOutSize !== n) { onnxOutBytes = new Uint8Array(n); onnxOutSize = n; }
  return onnxOutBytes;
}

// Per-model inference-time logging (first N frames). Helps spot silent
// fallbacks from WebGPU to WASM and confirm whether a "slow" model is
// slow because of heavy compute or because the EP didn't take.
const inferLogRemaining = { rvm: 3, u2netp: 3, silueta: 3, modnet: 3 };
function logInferTime(name, ms) {
  if (!(inferLogRemaining[name] > 0)) return;
  inferLogRemaining[name]--;
  console.log(`[${name}] inference ${ms.toFixed(0)} ms`);
}

// Generic non-recurrent ONNX segmenter. Resize → ImageNet normalize →
// session.run → minmax to [0,1] → uint8. Works for u2netp, silueta, and
// anything else in the U²-Net family that shipped as rembg.
async function simpleOnnxInfer(frame, cfg, session) {
  const W = cfg.w, H = cfg.h;
  ensureOnnxCanvas(W, H);
  onnxCtx.drawImage(frame, 0, 0, W, H);
  const d = onnxCtx.getImageData(0, 0, W, H).data;
  const n = W * H;
  const src = ensureSrcBuf(n);
  const [mr, mg, mb] = cfg.mean, [sr, sg, sb] = cfg.std;
  for (let i = 0; i < n; i++) {
    src[i]         = (d[i * 4]     / 255 - mr) / sr;  // R plane
    src[i + n]     = (d[i * 4 + 1] / 255 - mg) / sg;  // G plane
    src[i + 2 * n] = (d[i * 4 + 2] / 255 - mb) / sb;  // B plane
  }
  const srcT = new ortModule.Tensor('float32', src, [1, 3, H, W]);
  const inputName  = session.inputNames[0];
  const outputName = session.outputNames[0];
  const out = await session.run({ [inputName]: srcT }, {
    // Ensure the output is CPU-readable regardless of EP. Avoids any
    // stale-GPU-buffer pitfalls we already saw with RVM.
    preferredOutputLocation: 'cpu',
  });
  const outData = await out[outputName].getData();  // Float32Array, [1,1,H,W]
  const bytes = ensureOutBuf(outData.length);

  if (cfg.postProcess === 'sigmoid') {
    for (let i = 0; i < outData.length; i++) {
      const v = (1 / (1 + Math.exp(-outData[i]))) * 255;
      bytes[i] = v < 0 ? 0 : v > 255 ? 255 : v | 0;
    }
  } else if (cfg.postProcess === 'scale') {
    // Direct 0..1 → 0..255. Used for models (modnet) whose output is
    // already normalized; skips the two-pass min/max scan.
    for (let i = 0; i < outData.length; i++) {
      const v = outData[i] * 255;
      bytes[i] = v < 0 ? 0 : v > 255 ? 255 : v | 0;
    }
  } else {
    // minmax: rembg-style per-frame rescale.
    let omin = Infinity, omax = -Infinity;
    for (let i = 0; i < outData.length; i++) {
      if (outData[i] < omin) omin = outData[i];
      if (outData[i] > omax) omax = outData[i];
    }
    const range = Math.max(omax - omin, 1e-6);
    for (let i = 0; i < outData.length; i++) {
      const v = ((outData[i] - omin) / range) * 255;
      bytes[i] = v < 0 ? 0 : v > 255 ? 255 : v | 0;
    }
  }
  return { data: bytes, w: W, h: H };
}

async function ensureORT() {
  if (ortModule) return ortModule;
  // JSPI bundle: loads the new native WebGPU execution provider (complete
  // kernel library — fixes AveragePool ceil_mode which broke the legacy
  // JSEP path on RVM). Requires Chrome 137+. Falls back to WASM below.
  ortModule = await import('../ort/ort.jspi.min.mjs');
  // Point the WASM loader at our vendored /ort/ dir. Trailing slash matters.
  ortModule.env.wasm.wasmPaths = new URL('../ort/', import.meta.url).href;
  // Multithreaded WASM needs SharedArrayBuffer, which only exists when the page
  // is cross-origin-isolated (COOP + COEP). GH Pages can't set those headers;
  // local `serve` picks them up from docs/serve.json. Fall back to single-thread
  // when isolation isn't active so the worker still runs.
  const canThread = typeof SharedArrayBuffer !== 'undefined'
                 && (globalThis.crossOriginIsolated ?? true);
  ortModule.env.wasm.numThreads = canThread
    ? Math.min(navigator.hardwareConcurrency || 4, 8)
    : 1;
  return ortModule;
}

function resetRVMState() {
  const ort = ortModule;
  // ONNX-RVM accepts a [1,1,1,1] zero tensor for the initial recurrent state;
  // the model broadcasts it to the right shape internally on the first frame.
  const mk = () => new ort.Tensor('float32', new Float32Array([0]), [1, 1, 1, 1]);
  rvmState = { r1i: mk(), r2i: mk(), r3i: mk(), r4i: mk() };
}

async function loadOnnxModel(name) {
  const ort = await ensureORT();
  const url = onnxModelUrls[name];
  if (!url) throw new Error(`No URL configured for model "${name}"`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${name} fetch: ${res.status} ${res.statusText}`);
  const buf = new Uint8Array(await res.arrayBuffer());

  // RVM miscomputes on WebGPU (decoder head diverges); stay on WASM.
  // Simple feed-forward models (u2netp, silueta) are safe on WebGPU.
  const eps = SIMPLE_ONNX_MODELS[name]?.eps ?? ['wasm'];

  let session;
  try {
    session = await ort.InferenceSession.create(buf, {
      executionProviders: eps,
      graphOptimizationLevel: 'all',
    });
  } catch (err) {
    // If WebGPU can't build the session at all (missing op, no adapter),
    // retry on WASM rather than surface the error to the user.
    if (eps[0] === 'webgpu') {
      console.warn(`[${name}] WebGPU session init failed, falling back to WASM:`, err.message);
      session = await ort.InferenceSession.create(buf, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    } else {
      throw err;
    }
  }
  onnxSessions[name] = session;
  if (name === 'rvm') resetRVMState();
}

let rvmLoggedStats = false;

async function rvmInfer(frame) {
  const ort = ortModule;
  const rvmSession = onnxSessions.rvm;
  // Snapshot dims + generation once — quality can change mid-run; we want
  // tensor shapes to agree and to discard stale recurrent state on return.
  const { w: W, h: H } = onnxDims('rvm');
  const gen = rvmQualityGen;
  // Resize + read pixels via the shared OffscreenCanvas. VideoFrame is a
  // valid CanvasImageSource. getImageData returns RGBA uint8, which we
  // reshape into planar float32 CHW normalized to [0,1] as RVM expects.
  ensureOnnxCanvas(W, H);
  onnxCtx.drawImage(frame, 0, 0, W, H);
  const imgData = onnxCtx.getImageData(0, 0, W, H);
  const d = imgData.data;
  const n = W * H;
  const src = ensureSrcBuf(n);
  for (let i = 0; i < n; i++) {
    src[i]         = d[i * 4]     / 255;  // R plane
    src[i + n]     = d[i * 4 + 1] / 255;  // G plane
    src[i + 2 * n] = d[i * 4 + 2] / 255;  // B plane
  }
  const srcT = new ort.Tensor('float32', src, [1, 3, H, W]);
  // downsample_ratio is a rank-1 tensor in RVM's ONNX signature, not a scalar.
  // Per the RVM authors: 0.25 is for 1080p; 0.5 is recommended for SD input
  // like ours (512x288). Higher = sharper matte, slower inference.
  const dsr  = new ort.Tensor('float32', new Float32Array([0.5]), [1]);


  const out = await rvmSession.run({
    src: srcT,
    r1i: rvmState.r1i,
    r2i: rvmState.r2i,
    r3i: rvmState.r3i,
    r4i: rvmState.r4i,
    downsample_ratio: dsr,
  }, {
    // Force outputs to CPU so we don't feed potentially-aliased GPU buffers
    // back in as recurrent state on the next frame. Default is supposed to
    // be 'cpu' but being explicit here; this is the suspected root cause
    // of the state decaying to zero on the JSPI/WebGPU EP.
    preferredOutputLocation: 'cpu',
  });

  // If quality changed during this run, the returned r_o tensors are
  // shaped for the old input — throw them away, keep the fresh [1,1,1,1]
  // placeholders that setRvmQuality already installed, and skip returning
  // this stale matte.
  if (gen !== rvmQualityGen) return null;

  rvmState.r1i = out.r1o;
  rvmState.r2i = out.r2o;
  rvmState.r3i = out.r3o;
  rvmState.r4i = out.r4o;

  // On WebGPU, output tensors live on the GPU until explicitly read back.
  // Synchronous `.data` can return stale/empty data — always await getData().
  // On WASM the async call is effectively a no-op (data is already CPU-side).
  const pha = await out.pha.getData();  // Float32Array length H*W, alpha in [0,1]

  if (!rvmLoggedStats) {
    let mn = Infinity, mx = -Infinity, sum = 0;
    for (let i = 0; i < pha.length; i++) {
      if (pha[i] < mn) mn = pha[i];
      if (pha[i] > mx) mx = pha[i];
      sum += pha[i];
    }
    console.log(`[rvm] pha stats — min=${mn.toFixed(3)} max=${mx.toFixed(3)} mean=${(sum/pha.length).toFixed(3)} size=${W}x${H}`);
    rvmLoggedStats = true;
  }

  const bytes = ensureOutBuf(pha.length);
  for (let i = 0; i < pha.length; i++) {
    const v = pha[i] * 255;
    bytes[i] = v < 0 ? 0 : v > 255 ? 255 : v | 0;
  }
  return { data: bytes, w: W, h: H };
}

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
    'u_trim', 'u_maskInvert', 'u_maskMode',
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
  gl.uniform1f(u.u_trim,       params.trim);
  // Binary selfie_segmenter inverts foreground vs multiclass; flip the
  // shader's mask interpretation when it's the active model.
  gl.uniform1f(u.u_maskInvert, activeModel === 'binary' ? 1.0 : 0.0);
  // ONNX models (RVM, u2netp, silueta, modnet) write a continuous alpha
  // matte; MediaPipe writes a categorical mask.
  const continuous = activeModel === 'rvm' || SIMPLE_ONNX_MODELS[activeModel];
  gl.uniform1f(u.u_maskMode,   continuous ? 1.0 : 0.0);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}

// ---- Message handler --------------------------------------------------
self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'init') {
      initGL(msg.canvas);
      // All ONNX model URLs ride in on init. Each is lazy-loaded on first
      // setModel() to keep preload cheap for users who never switch off
      // MediaPipe.
      onnxModelUrls = msg.onnxModelUrls || {};

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
      activeModel         = 'binary';
      handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetBuffer: msg.handBuffer, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 2,
      });
      self.postMessage({ type: 'ready' });
      return;
    }

    if (msg.type === 'setNumHands') {
      const n = Math.max(1, Math.min(4, msg.numHands | 0));
      if (handLandmarker) {
        await handLandmarker.setOptions({ numHands: n });
      }
      return;
    }

    if (msg.type === 'setRvmQuality') {
      if (!['quality', 'balanced', 'fast'].includes(msg.quality)) return;
      onnxQuality = msg.quality;
      rvmQualityGen++;
      // Clear RVM recurrent state (its tensor shape depends on input size).
      if (onnxSessions.rvm && ortModule) resetRVMState();
      // Reset inference-time logs so user can compare old vs new preset.
      for (const k of Object.keys(inferLogRemaining)) inferLogRemaining[k] = 3;
      return;
    }

    if (msg.type === 'setModel') {
      const name = msg.model;
      // ONNX-backed models: rvm / u2netp / silueta. Shared lazy-load with
      // status messages so the UI can tell the user a first-time download
      // is happening.
      if (name === 'rvm' || name === 'u2netp' || name === 'silueta' || name === 'modnet') {
        if (!onnxSessions[name] && !onnxLoading[name]) {
          onnxLoading[name] = true;
          self.postMessage({ type: 'rvmStatus', status: 'loading', model: name });
          try {
            await loadOnnxModel(name);
          } catch (err) {
            onnxLoading[name] = false;
            self.postMessage({ type: 'rvmStatus', status: 'error', model: name, message: err.message });
            return;
          }
          onnxLoading[name] = false;
          self.postMessage({ type: 'rvmStatus', status: 'ready', model: name });
        }
        if (onnxSessions[name]) {
          activeModel     = name;
          activeSegmenter = null;
          // Reset RVM's recurrent state when entering that model so the
          // first matte isn't seeded with stale features.
          if (name === 'rvm') resetRVMState();
        }
      } else {
        activeModel     = name;  // 'binary' | 'multiclass'
        activeSegmenter = name === 'multiclass' ? segmenterMulticlass : segmenterBinary;
      }
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

      const wantSeg   = !!msg.wantSeg;
      const wantHands = !!msg.wantHands;

      // ONNX inference runs in parallel with the render path. The sync
      // preamble (drawImage + getImageData + tensor build) completes before
      // the function hits its first await, so the frame is safe to close
      // immediately after. The returned promise finishes later and uploads
      // the new mask; the next render tick picks it up.
      //
      // MediaPipe paths (activeSegmenter, handLandmarker) are synchronous and
      // still run before draw so the fresh mask lands on the current frame.
      let segPromise = null;

      if (wantSeg) {
        if (activeModel === 'rvm' && onnxSessions.rvm) {
          const t0 = performance.now();
          segPromise = rvmInfer(frame).then(result => {
            logInferTime('rvm', performance.now() - t0);
            if (result) uploadMask(result.data, result.w, result.h);
          });
        } else if (SIMPLE_ONNX_MODELS[activeModel] && onnxSessions[activeModel]) {
          const cfg = { ...SIMPLE_ONNX_MODELS[activeModel], ...onnxDims(activeModel) };
          const t0 = performance.now();
          segPromise = simpleOnnxInfer(frame, cfg, onnxSessions[activeModel]).then(result => {
            logInferTime(activeModel, performance.now() - t0);
            uploadMask(result.data, result.w, result.h);
          });
        } else if (activeSegmenter) {
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
      }

      let landmarks = null;
      if (wantHands) {
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
      });

      // Signal inference completion separately so main can release its
      // inference gate independently of the render gate. For MediaPipe /
      // hands-only the work is already done; post immediately.
      if (wantSeg || wantHands) {
        if (segPromise) {
          segPromise.then(() => {
            self.postMessage({ type: 'inferenceDone', ts: msg.ts });
          }).catch(err => {
            self.postMessage({
              type: 'error',
              stage: 'inference',
              message: err?.message || String(err),
              stack: err?.stack,
            });
          });
        } else {
          self.postMessage({ type: 'inferenceDone', ts: msg.ts });
        }
      }
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
