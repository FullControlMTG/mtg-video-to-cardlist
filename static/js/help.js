// Help modal: open/close behaviour for the ? button in the header.

import { $ } from './dom.js';

const overlay = $('help-overlay');

$('help-btn').addEventListener('click', () => overlay.classList.add('open'));
$('help-close').addEventListener('click', () => overlay.classList.remove('open'));
overlay.addEventListener('click', e => {
  if (e.target === overlay) overlay.classList.remove('open');
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && overlay.classList.contains('open')) overlay.classList.remove('open');
});
