// Layout state: sidebar collapse/expand + left-pane stacked/side-by-side.
// Both are pure class toggles; CSS in layout.css/deck.css drives the rest.
// The left-pane orientation persists via localStorage so it survives
// reloads, since the choice is user preference (portrait camera vs.
// landscape stacked works differently for different people).

import { $ } from './dom.js';

const layoutEl = $('layout');
const paneLeft = document.querySelector('.pane-left');
const modeBtn  = $('layout-mode-btn');

const SIDE_BY_SIDE_KEY = 'layout.side-by-side';

// ── Sidebar collapse ────────────────────────────────────────────────
$('toggle-deck-btn').addEventListener('click', () => layoutEl.classList.add('deck-hidden'));
$('show-deck-btn').addEventListener('click',   () => layoutEl.classList.remove('deck-hidden'));

// ── Left-pane stacked / side-by-side ────────────────────────────────
function applyLayoutMode(sideBySide) {
  paneLeft.classList.toggle('side-by-side', sideBySide);
  modeBtn.title = sideBySide
    ? 'Switch to stacked layout'
    : 'Switch to side-by-side layout';
  modeBtn.setAttribute('aria-pressed', String(sideBySide));
}

modeBtn.addEventListener('click', () => {
  const next = !paneLeft.classList.contains('side-by-side');
  applyLayoutMode(next);
  try { localStorage.setItem(SIDE_BY_SIDE_KEY, next ? '1' : '0'); } catch { /* private mode */ }
});

// Restore saved preference on load.
try {
  applyLayoutMode(localStorage.getItem(SIDE_BY_SIDE_KEY) === '1');
} catch {
  applyLayoutMode(false);
}
