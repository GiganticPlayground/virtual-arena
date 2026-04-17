// GLSL 300 es shader source for the compositor.

export const VS = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

// Fragment pipeline:
//   1. Sample the arena background at "cover-fit" UVs (no distortion).
//   2. Map canvas UV -> source UV with user-controlled size + offset;
//      `inside` is 1 where the source covers this pixel, else bg shows.
//   3. 3x3 tap of the category mask, averaged and smoothstepped.
//   4. Mix bg/cam by that alpha, gated by `inside`.
export const FS = `#version 300 es
precision highp float;
in vec2 v_uv;
out vec4 outColor;

uniform sampler2D u_bg;
uniform sampler2D u_cam;
uniform sampler2D u_mask;
uniform float u_feather;   // feather radius in mask-texel units
uniform vec2  u_maskSize;  // e.g. (256, 256)
uniform float u_mirror;    // 1.0 -> flip camera/mask x
uniform vec2  u_bgScale;   // cover-fit scale
uniform vec2  u_bgOffset;  // cover-fit offset
uniform float u_camScale;  // source size (1.0 = fill canvas)
uniform vec2  u_camOffset; // source position in canvas UV

float sampleMask(vec2 uv) {
  // multiclass: any category != 0 is "person".
  float v = texture(u_mask, uv).r;
  return step(0.5 / 255.0, v);
}

void main() {
  vec2 bgUv = clamp(v_uv * u_bgScale + u_bgOffset, 0.0, 1.0);
  vec4 bg = texture(u_bg, bgUv);

  vec2 camUv = (v_uv - u_camOffset) / u_camScale;
  float inside = step(0.0, camUv.x) * step(camUv.x, 1.0)
               * step(0.0, camUv.y) * step(camUv.y, 1.0);

  if (u_mirror > 0.5) camUv.x = 1.0 - camUv.x;
  vec4 cam = texture(u_cam, camUv);

  vec2 texel = 1.0 / u_maskSize;
  float step_ = max(u_feather, 0.0001);
  float m = 0.0;
  for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
      vec2 ofs = vec2(float(dx), float(dy)) * texel * step_;
      m += sampleMask(camUv + ofs);
    }
  }
  m /= 9.0;

  float alpha = smoothstep(0.35, 0.65, m) * inside;
  outColor = vec4(mix(bg.rgb, cam.rgb, alpha), 1.0);
}`;
