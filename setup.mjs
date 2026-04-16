// Offline-asset setup: downloads arena images + the selfie_multiclass model
// and copies the MediaPipe tasks-vision bundle + wasm out of node_modules
// into docs/ so the prototype runs with zero external requests at runtime.
// (docs/ is what GitHub Pages serves from when "Branch: main / docs" is set.)
//
// Run with: `node setup.mjs` (or `npm run setup`).
// Idempotent: anything already present is skipped.

import { mkdir, cp, writeFile, access, stat } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const PUB = join(__dirname, 'docs');

const ARENAS = [
  { name: 'arena-wide.jpg',   url: 'https://images.unsplash.com/photo-1504450758481-7338eba7524a?auto=format&fit=crop&w=1920&q=80' },
  { name: 'court-low.jpg',    url: 'https://images.unsplash.com/photo-1546519638-68e109498ffc?auto=format&fit=crop&w=1920&q=80' },
  { name: 'crowd-lights.jpg', url: 'https://images.unsplash.com/photo-1518091043644-c1d4457512c6?auto=format&fit=crop&w=1920&q=80' },
];

const MODELS = [
  {
    name: 'selfie_multiclass_256x256.tflite',
    url:  'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_multiclass_256x256/float32/latest/selfie_multiclass_256x256.tflite',
  },
  {
    name: 'hand_landmarker.task',
    url:  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task',
  },
];

async function exists(p) {
  try { await access(p); return true; } catch { return false; }
}

async function download(url, dest) {
  if (await exists(dest)) {
    const s = await stat(dest);
    console.log(`  skip  ${dest}  (${(s.size/1024).toFixed(0)} KB already present)`);
    return;
  }
  process.stdout.write(`  fetch ${url} ... `);
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await writeFile(dest, buf);
  console.log(`${(buf.byteLength/1024).toFixed(0)} KB -> ${dest}`);
}

async function copyIfMissing(src, dst, label) {
  if (await exists(dst)) {
    console.log(`  skip  ${dst}  (already present)`);
    return;
  }
  await cp(src, dst, { recursive: true });
  console.log(`  copy  ${label}  -> ${dst}`);
}

async function main() {
  await mkdir(join(PUB, 'arenas'), { recursive: true });
  await mkdir(join(PUB, 'models'), { recursive: true });
  await mkdir(join(PUB, 'vision'), { recursive: true });

  console.log('[1/3] Arena images (Unsplash)');
  for (const a of ARENAS) {
    await download(a.url, join(PUB, 'arenas', a.name));
  }

  console.log('[2/3] MediaPipe models');
  for (const m of MODELS) {
    await download(m.url, join(PUB, 'models', m.name));
  }

  console.log('[3/3] MediaPipe tasks-vision bundle + wasm');
  const pkg = join(__dirname, 'node_modules', '@mediapipe', 'tasks-vision');
  if (!(await exists(pkg))) {
    throw new Error('node_modules/@mediapipe/tasks-vision missing — run `npm install` first.');
  }
  await copyIfMissing(join(pkg, 'vision_bundle.mjs'), join(PUB, 'vision', 'vision_bundle.mjs'), 'vision_bundle.mjs');
  await copyIfMissing(join(pkg, 'wasm'), join(PUB, 'vision', 'wasm'), 'wasm/');

  console.log('\nSetup complete. Run `npm start` to serve on http://localhost:8123');
}

main().catch(e => {
  console.error('\nSetup failed:', e.message);
  process.exit(1);
});
