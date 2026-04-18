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
uniform float u_trim;      // silhouette trim: >0 erodes, <0 dilates
uniform float u_maskInvert; // 1.0 when the active segmenter uses 0=person

float sampleMask(vec2 uv) {
  // Multiclass model: any non-zero category is "person".
  // Binary selfie_segmenter: category 0 is "person" — inverted. The worker
  // sets u_maskInvert based on the active model so this is transparent here.
  float v = texture(u_mask, uv).r;
  float fg = step(0.5 / 255.0, v);
  return u_maskInvert > 0.5 ? (1.0 - fg) : fg;
}

void main() {
  vec2 bgUv = clamp(v_uv * u_bgScale + u_bgOffset, 0.0, 1.0);
  vec4 bg = texture(u_bg, bgUv);

  vec2 camUv = (v_uv - u_camOffset) / u_camScale;
  float inside = step(0.0, camUv.x) * step(camUv.x, 1.0)
               * step(0.0, camUv.y) * step(camUv.y, 1.0);

  if (u_mirror > 0.5) camUv.x = 1.0 - camUv.x;
  vec4 cam = texture(u_cam, camUv);

  // 5x5 mask tap kernel. Outer taps (dx/dy = ±2) sit ~u_feather mask
  // texels from center; inner taps interpolate in between. 25 taps gives
  // 26 discrete m levels, smooth enough for u_feather to visibly widen
  // the soft silhouette edge from hard (feather~0) to very soft (feather=5).
  vec2 texel = 1.0 / u_maskSize;
  float scale = max(u_feather, 0.0001) * 0.5;
  float m = 0.0;
  for (int dy = -2; dy <= 2; ++dy) {
    for (int dx = -2; dx <= 2; ++dx) {
      vec2 ofs = vec2(float(dx), float(dy)) * texel * scale;
      m += sampleMask(camUv + ofs);
    }
  }
  m /= 25.0;

  // Trim shifts the smoothstep band. Positive trim requires more of the
  // kernel to be foreground before outputting alpha, which peels off thin
  // bleed (chair backs, armrests) at the cost of slightly thinner edges.
  // Negative trim dilates — useful for catching hair wisps.
  float lo = clamp(0.35 + u_trim, 0.0, 0.98);
  float hi = clamp(0.65 + u_trim, lo + 0.02, 1.0);
  float alpha = smoothstep(lo, hi, m) * inside;
  outColor = vec4(mix(bg.rgb, cam.rgb, alpha), 1.0);
}`;
