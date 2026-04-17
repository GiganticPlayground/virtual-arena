// Video source lifecycle: webcam + video file. Fully detaches whatever is
// attached before starting a new source.

import { video, sourceSel } from './dom.js';

let stream = null;

export function teardownSource() {
  if (stream) {
    for (const t of stream.getTracks()) t.stop();
    stream = null;
  }
  video.srcObject = null;
  video.removeAttribute('src');
  video.load();
  video.loop = false;
}

export async function startSource() {
  teardownSource();
  const src = sourceSel.value;
  if (src === 'webcam') {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width:  { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: 'user',
        },
        audio: false,
      });
    } catch (e) {
      // Caller surfaces this to the UI.
      throw new Error('Webcam unavailable or permission denied. Allow camera access and retry.');
    }
    video.srcObject = stream;
  } else {
    // Looped file playback. The <video> has the `muted` attribute in HTML
    // so autoplay is allowed.
    video.src = src;
    video.loop = true;
  }
  await new Promise((res, rej) => {
    video.addEventListener('loadedmetadata', res, { once: true });
    video.addEventListener('error', () =>
      rej(new Error(`Failed to load video source: ${src}`)), { once: true });
  });
  await video.play();
}
