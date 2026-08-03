// "Detected This Session" panel on the left pane. Receives card names
// from the WS detected stream, looks up their Scryfall image/data, and
// renders a clickable thumbnail that adds the card to the active zone.

import { $, escHtml } from './dom.js';
import { state } from './state.js';
import { api } from './api.js';
import { addCard, addManyOnce } from './deck.js';

// If you don't see this line in the console on page load, your browser
// is serving a cached older detected.js — Add All won't have its click
// handler bound. Bust cache (DevTools → Network → Disable cache → reload).
console.log('[detected.js] loaded with add-all support');

const detectedGrid = $('detected-grid');
const scanBadge    = $('scan-badge');
const addAllBtn    = $('add-all-detected-btn');

const EMPTY_HINT = '<p class="empty-hint">Cards recognised by the camera will appear here.</p>';

// Update the scan badge text and toggle the pulse animation. Pulse only
// while we're actively awaiting a scan (empty grid); once cards land the
// badge shows a stable count. Also gates the "Add All" button — no cards,
// nothing to add.
function updateScanBadge() {
  const n = state.detectedCards.size;
  scanBadge.textContent = n ? `${n} found` : 'scanning…';
  scanBadge.classList.toggle('pulsing', n === 0);
  addAllBtn.disabled = n === 0;
}

export async function handleDetected(cards) {
  for (const c of cards) {
    if (!c.name || state.detectedCards.has(c.name)) continue;

    // Reserve the slot so two near-simultaneous WS messages don't double-render.
    state.detectedCards.set(c.name, { name: c.name });

    const cardData = (await api.getCard(c.name)) || { name: c.name };
    state.detectedCards.set(c.name, cardData);
    renderDetectedCard(c.name, cardData);
  }

  updateScanBadge();
}

function renderDetectedCard(name, cardData) {
  const hint = detectedGrid.querySelector('.empty-hint');
  if (hint) hint.remove();

  // Only auto-scroll if the user was already near the bottom — don't yank
  // them away from an earlier card they might be inspecting.
  const wasAtBottom =
    detectedGrid.scrollHeight - detectedGrid.scrollTop - detectedGrid.clientHeight < 40;

  const img = cardData?.image_uri || '';
  const div = document.createElement('div');
  div.className = 'detected-card';
  div.dataset.name = name;
  // role/tabindex so keyboard users can add via Enter/Space on the card.
  div.setAttribute('role', 'button');
  div.setAttribute('tabindex', '0');
  div.setAttribute('aria-label', `Add ${name} to deck`);
  div.innerHTML = `
    <img src="${escHtml(img)}" alt="${escHtml(name)}" loading="lazy"
         onerror="this.style.visibility='hidden'" />
    <div class="card-label">${escHtml(name)}</div>
    <button class="detected-card-dismiss" aria-label="Dismiss ${escHtml(name)}">&times;</button>
    <div class="add-overlay">
      <button class="add-overlay-btn" aria-label="Add ${escHtml(name)} to deck">+ Add</button>
    </div>
  `;

  const dismissBtn = div.querySelector('.detected-card-dismiss');
  dismissBtn.addEventListener('click', e => {
    e.stopPropagation();
    state.detectedCards.delete(name);
    div.remove();
    if (!detectedGrid.querySelector('.detected-card')) detectedGrid.innerHTML = EMPTY_HINT;
    updateScanBadge();
  });

  const addBtn = div.querySelector('.add-overlay-btn');
  const doAdd = async e => {
    e.stopPropagation();
    await addCard(name, 1, state.activeZone);
    addBtn.textContent = '✓ Added';
    setTimeout(() => { addBtn.textContent = '+ Add'; }, 1200);
  };
  addBtn.addEventListener('click', doAdd);
  div.addEventListener('click', e => {
    if (e.target === dismissBtn || e.target === addBtn) return;
    doAdd(e);
  });
  div.addEventListener('keydown', e => {
    // Only handle keys on the card itself; inner buttons manage their own.
    if (e.target !== div) return;
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      doAdd(e);
    }
  });

  detectedGrid.appendChild(div);

  if (wasAtBottom) detectedGrid.scrollTop = detectedGrid.scrollHeight;
}

$('clear-detected-btn').addEventListener('click', () => {
  state.detectedCards.clear();
  detectedGrid.innerHTML = EMPTY_HINT;
  updateScanBadge();
});

// Add-all: one copy of every detected card, in parallel, into the active
// zone. Common flow for singleton decks — scan a batch, click once.
// Marks each corresponding tile with a "✓ added" state so the user can
// tell what landed vs. what came in after the click; doesn't clear the
// panel automatically (that's what Clear All is for).
addAllBtn.addEventListener('click', async () => {
  const names = [...state.detectedCards.keys()];
  console.log('[add-all] click; detectedCards=', names, 'zone=', state.activeZone);
  if (!names.length) return;

  addAllBtn.disabled = true;
  const originalLabel = addAllBtn.textContent;
  addAllBtn.textContent = `Adding ${names.length}…`;

  // Walk the grid once and index tiles by their data-name — safer than
  // building a querySelector from arbitrary card names, which can produce
  // unusable selectors for names with quotes / commas / punctuation.
  const tilesAtClick = new Map();
  for (const tile of detectedGrid.querySelectorAll('.detected-card')) {
    const n = tile.dataset.name;
    if (state.detectedCards.has(n)) {
      tilesAtClick.set(n, tile);
      tile.classList.add('is-adding');
    }
  }

  let ok = 0, failed = [];
  try {
    const res = await addManyOnce(names, state.activeZone, 1);
    ok = res.ok;
    failed = res.failed;
  } catch (err) {
    console.error('[add-all] addManyOnce threw:', err);
    failed = names.slice();
  }

  for (const [n, tile] of tilesAtClick) {
    tile.classList.remove('is-adding');
    if (!failed.includes(n)) tile.classList.add('is-added');
  }

  addAllBtn.textContent = failed.length
    ? `Added ${ok}, ${failed.length} failed`
    : `✓ Added ${ok}`;
  setTimeout(() => {
    addAllBtn.textContent = originalLabel;
    updateScanBadge();  // re-enables if there are still cards left
  }, 1400);
});
