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

// Robust Video Matting (MobileNetV3 FP32, ~15 MB) via onnxruntime-web.
// Third optional model; produces a continuous alpha matte instead of a
// categorical mask, so the shader takes a different path when it's active.
export const RVM_MODEL_URL      = './models/rvm_mobilenetv3_fp32.onnx';
// U²-Net-p: tiny (4.5 MB) salient-object segmenter from the rembg project.
// Non-recurrent, 320x320 input. Quick to try but coarser matte than RVM.
export const U2NETP_MODEL_URL   = './models/u2netp.onnx';
// Silueta: silhouette-trained saliency model (~42 MB) from rembg. Slower
// than u2netp but much crisper edges and handles multi-subject OK.
export const SILUETA_MODEL_URL  = './models/silueta.onnx';
// MODNet (portrait matting, ~26 MB) from Xenova/modnet on HuggingFace.
// Simpler op set than Silueta — WebGPU carries it at interactive rates.
export const MODNET_MODEL_URL   = './models/modnet.onnx';
export const ORT_URL            = './ort/';

// Foam finger art: PNG (1024x1536, 2:3) drawn pointing up; rotated at
// runtime to match the hand orientation.
export const FOAM_IMG_URL = './foam_finger.png';
