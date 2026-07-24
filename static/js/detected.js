// "Detected This Session" panel on the left pane. Receives card names
// from the WS detected stream, looks up their Scryfall image/data, and
// renders a clickable thumbnail that adds the card to the active zone.

import { $, escHtml } from './dom.js';
import { state } from './state.js';
import { api } from './api.js';
import { addCard } from './deck.js';

const detectedGrid = $('detected-grid');
const scanBadge    = $('scan-badge');

const EMPTY_HINT = '<p class="empty-hint">Cards recognised by the camera will appear here.</p>';

export async function handleDetected(cards) {
  for (const c of cards) {
    if (!c.name || state.detectedCards.has(c.name)) continue;

    // Reserve the slot so two near-simultaneous WS messages don't double-render.
    state.detectedCards.set(c.name, { name: c.name });

    const cardData = (await api.getCard(c.name)) || { name: c.name };
    state.detectedCards.set(c.name, cardData);
    renderDetectedCard(c.name, cardData);
  }

  const n = state.detectedCards.size;
  scanBadge.textContent = n ? `${n} found` : 'scanning…';
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
    scanBadge.textContent = state.detectedCards.size ? `${state.detectedCards.size} found` : 'scanning…';
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
  scanBadge.textContent = 'scanning…';
});
