// WebGL2 compositor. Owns the GL context, shader program, uniform handles,
// and the three textures (bg/cam/mask). Exposes upload helpers + draw().
// No render state leaks out of this module — drawFrame() takes everything
// it needs as arguments.

import { canvas, showError } from './dom.js';
import { VS, FS } from './shaders.js';

export const gl = canvas.getContext('webgl2', {
  alpha: false, antialias: false, premultipliedAlpha: false,
});
if (!gl) {
  showError('WebGL2 unavailable. This prototype targets Chrome desktop with WebGL2 support.');
  throw new Error('webgl2-unavailable');
}

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error('Shader compile: ' + gl.getShaderInfoLog(s));
  }
  return s;
}

const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog);
if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
  throw new Error('Program link: ' + gl.getProgramInfoLog(prog));
}
gl.useProgram(prog);

// Fullscreen quad.
const vao = gl.createVertexArray();
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

const u = Object.fromEntries([
  'u_bg', 'u_cam', 'u_mask',
  'u_feather', 'u_maskSize', 'u_mirror',
  'u_bgScale', 'u_bgOffset',
  'u_camScale', 'u_camOffset',
].map(n => [n, gl.getUniformLocation(prog, n)]));

function makeTex() {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return t;
}
const bgTex   = makeTex();
const camTex  = makeTex();
const maskTex = makeTex();

// Upload an RGB image-like (HTMLImageElement/HTMLVideoElement/ImageBitmap)
// into the background texture. Returns the image's aspect ratio so the
// caller can cover-fit.
export function uploadBackgroundImage(img) {
  gl.bindTexture(gl.TEXTURE_2D, bgTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
  return img.naturalWidth / img.naturalHeight;
}

export function uploadCamFrame(videoEl) {
  gl.bindTexture(gl.TEXTURE_2D, camTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, videoEl);
}

export function uploadMask(mask, width, height) {
  gl.bindTexture(gl.TEXTURE_2D, maskTex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, width, height, 0, gl.RED, gl.UNSIGNED_BYTE, mask);
  gl.uniform2f(u.u_maskSize, width, height);
}

// Draw one frame. bgAspect lets this module own the cover-fit math without
// reaching into the background cache.
export function drawFrame({ bgAspect, feather, mirror, camScale, camOffsetX, camOffsetY }) {
  const canvasAspect = canvas.width / canvas.height;
  let sx = 1, sy = 1, ox = 0, oy = 0;
  if (bgAspect > canvasAspect) { sx = canvasAspect / bgAspect; ox = (1 - sx) * 0.5; }
  else                         { sy = bgAspect / canvasAspect; oy = (1 - sy) * 0.5; }

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, bgTex);   gl.uniform1i(u.u_bg,   0);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, camTex);  gl.uniform1i(u.u_cam,  1);
  gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, maskTex); gl.uniform1i(u.u_mask, 2);
  gl.uniform1f(u.u_feather,  feather);
  gl.uniform1f(u.u_mirror,   mirror ? 1.0 : 0.0);
  gl.uniform2f(u.u_bgScale,  sx, sy);
  gl.uniform2f(u.u_bgOffset, ox, oy);
  gl.uniform1f(u.u_camScale, camScale);
  gl.uniform2f(u.u_camOffset, camOffsetX, camOffsetY);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}
