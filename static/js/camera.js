// Camera: device dropdown, 90° rotate button, lifecycle-state badge,
// and the video-feed error/reload helper.

import { $ } from './dom.js';
import { api } from './api.js';

const cameraSelect = $('camera-select');
const camStatus    = $('cam-status');
const rotateBtn    = $('rotate-btn');
const videoFeed    = $('video-feed');

let _cameraRotation = 0;

function reloadVideoFeed() {
  videoFeed.src = '';
  setTimeout(() => { videoFeed.src = '/video'; }, 400);
}

videoFeed.addEventListener('error', () => {
  setTimeout(reloadVideoFeed, 2000);
});

cameraSelect.addEventListener('change', async () => {
  const source = parseInt(cameraSelect.value, 10);
  if (isNaN(source)) return;
  await api.selectCamera(source);
  reloadVideoFeed();
});

rotateBtn.addEventListener('click', async () => {
  _cameraRotation = (_cameraRotation + 90) % 360;
  // No reloadVideoFeed() — the MJPEG stream is continuous; the server
  // starts emitting rotated frames immediately.
  await api.rotateCamera(_cameraRotation);
});

export async function loadCameras() {
  const data = await api.getCameras();
  if (!data) return;
  const { cameras, current, rotation } = data;

  cameraSelect.innerHTML = cameras.length
    ? cameras.map(c => `<option value="${c.index}">${c.name} (${c.resolution})</option>`).join('')
    : '<option value="">No cameras found</option>';

  cameraSelect.value = String(current);
  if (rotation !== undefined) _cameraRotation = rotation;
}

// Poll the camera lifecycle state so the user can see whether we're
// actually receiving frames (the app is useless if we can't), without
// re-enumerating devices on every poll.
export async function pollCameraStatus() {
  const s = await api.getCameraStatus();
  if (!s) return;
  camStatus.textContent = `camera: ${s.state}`;
  camStatus.classList.toggle('badge-ok',   s.state === 'streaming');
  camStatus.classList.toggle('badge-warn',
      s.state === 'connecting' || s.state === 'warming_up' || s.state === 'reconnecting');
  camStatus.classList.toggle('badge-err',  s.state === 'failed');
}
