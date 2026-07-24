// Decklist sidebar: renders the main/sideboard rows, wires the
// per-row controls (minus, set count, remove all), the zone tabs at
// the top, the global "Clear Deck" button, and the addCard() helper
// used by both search and detected panels.

import { $, escHtml } from './dom.js';
import { state } from './state.js';
import { api } from './api.js';
import { openCardModal } from './modal.js';
import { promptNumber } from './prompt.js';

const deckMain  = $('decklist-main');
const deckSide  = $('decklist-side');
const mainCount = $('main-count');
const sideCount = $('side-count');

export async function loadDeck() {
  const data = await api.getDeck();
  if (data) applyDeckUpdate(data);
}

export function applyDeckUpdate(data) {
  state.deck = data;
  renderZone('main', data.main || []);
  renderZone('side', data.side || []);
  mainCount.textContent = data.total_main || 0;
  sideCount.textContent = data.total_side || 0;
}

export async function addCard(name, count = 1, zone = 'main') {
  try {
    const resp = await api.addCard(name, count, zone);
    if (!resp.ok) {
      console.error('Add card error:', await resp.text());
      return;
    }
    // Refresh directly so the UI is consistent even if the WS update is delayed.
    await loadDeck();
  } catch (err) {
    console.error('Add card failed:', err);
  }
}

function renderZone(zone, entries) {
  const el = zone === 'main' ? deckMain : deckSide;

  if (!entries.length) {
    el.innerHTML = '<div class="deck-empty">No cards yet.</div>';
    return;
  }

  el.innerHTML = entries.map(e => `
    <div class="deck-row"
         role="button"
         tabindex="0"
         aria-label="Show details for ${escHtml(e.name)}"
         data-name="${escHtml(e.name)}"
         data-zone="${zone}">
      <img class="deck-row-thumb"
           src="${escHtml(e.image_uri || '')}"
           alt="${escHtml(e.name)}"
           onerror="this.style.visibility='hidden'" />
      <div>
        <div class="deck-row-name">${escHtml(e.name)}</div>
        <div class="deck-row-mana">${escHtml(e.mana_cost || '')}</div>
      </div>
      <div class="deck-row-controls">
        ${e.count > 1 ? `<button class="deck-row-decrement"
                                  title="Remove one copy"
                                  data-name="${escHtml(e.name)}"
                                  data-zone="${zone}">&minus;</button>` : ''}
        <button class="deck-row-count"
                title="Click to set count"
                data-name="${escHtml(e.name)}"
                data-zone="${zone}"
                data-count="${e.count}">${e.count}</button>
      </div>
      <button class="deck-row-remove"
              title="Remove all copies"
              data-name="${escHtml(e.name)}"
              data-zone="${zone}">×</button>
    </div>
  `).join('');

  // Row body → open detail modal. Skip clicks on the controls cluster
  // (minus/set-count) and the X remove button — those have their own handlers.
  el.querySelectorAll('.deck-row').forEach(row => {
    const openDetail = () => {
      const entry = entries.find(x => x.name === row.dataset.name);
      if (entry) openCardModal(entry);
    };
    row.addEventListener('click', e => {
      if (e.target.closest('.deck-row-controls') ||
          e.target.classList.contains('deck-row-remove')) return;
      openDetail();
    });
    row.addEventListener('keydown', e => {
      // Row is role=button — Enter/Space should activate it. Ignore when
      // focus is on a nested control (the inner buttons handle their own keys).
      if (e.target !== row) return;
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        openDetail();
      }
    });
  });

  el.querySelectorAll('.deck-row-decrement').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.stopPropagation();
      await api.decrementOne(btn.dataset.name, btn.dataset.zone);
      await loadDeck();
    });
  });

  el.querySelectorAll('.deck-row-count').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.stopPropagation();
      const { name, zone: z } = btn.dataset;
      const current = parseInt(btn.dataset.count, 10) || 1;
      const next = await promptNumber({
        title: 'Set count',
        label: name,
        value: current,
        min: 0,
        max: 99,
      });
      if (next == null || next === current) return;
      await api.setCount(name, next, z);
      await loadDeck();
    });
  });

  el.querySelectorAll('.deck-row-remove').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.stopPropagation();
      // PATCH count=0 deletes the entry server-side (set_count pops at 0).
      await api.setCount(btn.dataset.name, 0, btn.dataset.zone);
      await loadDeck();
    });
  });
}

// Zone tabs (Main / Sideboard) at the top of the sidebar.
document.querySelectorAll('.zone-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.zone-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    state.activeZone = tab.dataset.zone;
    deckMain.hidden = state.activeZone !== 'main';
    deckSide.hidden = state.activeZone !== 'side';
  });
});

// Clear-deck button at the bottom of the export bar.
$('clear-deck-btn').addEventListener('click', async () => {
  if (!confirm('Clear the entire decklist? This cannot be undone.')) return;
  await api.clearDeck();
  await loadDeck();
});
