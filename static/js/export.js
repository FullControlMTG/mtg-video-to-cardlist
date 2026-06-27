// Export modal: triggered by the format buttons at the bottom of the
// sidebar. Fetches the formatted text, fills the textarea, hides the
// Download .txt button for non-text formats (where the file would be
// renamed away from .txt anyway).

import { $ } from './dom.js';
import { api } from './api.js';

const exportOverlay = $('export-overlay');
const exportTitle   = $('export-title');
const exportText    = $('export-text');
const copyBtn       = $('copy-btn');
const downloadBtn   = $('download-btn');

document.querySelectorAll('.btn-export').forEach(btn => {
  btn.addEventListener('click', async () => {
    const fmt = btn.dataset.fmt;
    const text = await api.exportDeck(fmt);
    if (text == null) return;
    exportTitle.textContent = `Export – ${fmt.toUpperCase()}`;
    exportText.value = text;
    downloadBtn.style.display = fmt === 'text' ? '' : 'none';
    exportOverlay.classList.add('open');
  });
});

$('export-close').addEventListener('click', () => exportOverlay.classList.remove('open'));
exportOverlay.addEventListener('click', e => {
  if (e.target === exportOverlay) exportOverlay.classList.remove('open');
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && exportOverlay.classList.contains('open'))
    exportOverlay.classList.remove('open');
});

copyBtn.addEventListener('click', () => {
  navigator.clipboard.writeText(exportText.value).then(() => {
    copyBtn.textContent = 'Copied!';
    setTimeout(() => { copyBtn.textContent = 'Copy to Clipboard'; }, 1800);
  });
});

downloadBtn.addEventListener('click', () => {
  const blob = new Blob([exportText.value], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'decklist.txt';
  a.click();
  URL.revokeObjectURL(a.href);
});
