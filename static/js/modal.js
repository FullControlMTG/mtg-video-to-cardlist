// Card-detail modal: shown when a card is clicked in the deck, the
// search dropdown, or directly via openCardModal().

import { $, escHtml } from './dom.js';
import { state } from './state.js';
import { addCard } from './deck.js';

const modalOverlay = $('modal-overlay');
const modalBody    = $('modal-body');

export function openCardModal(cardData) {
  modalBody.innerHTML = `
    <img class="modal-card-img"
         src="${escHtml(cardData.image_uri || '')}"
         alt="${escHtml(cardData.name)}"
         onerror="this.style.display='none'" />
    <div class="modal-card-info">
      <div class="modal-card-name">${escHtml(cardData.name)}</div>
      <div class="modal-card-type">
        ${escHtml(cardData.mana_cost || '')}
        ${cardData.mana_cost && cardData.type_line ? '&nbsp;·&nbsp;' : ''}
        ${escHtml(cardData.type_line || '')}
      </div>
      <div class="modal-card-text">${escHtml(cardData.oracle_text || '')}</div>
      <div class="modal-add-controls">
        <input id="modal-count" class="modal-count-input"
               type="number" value="1" min="1" max="99" />
        <select id="modal-zone" class="modal-zone-select">
          <option value="main">Main Deck</option>
          <option value="side">Sideboard</option>
        </select>
        <button class="modal-add-btn" id="modal-add-btn">Add to Deck</button>
      </div>
    </div>
  `;

  $('modal-zone').value = state.activeZone;
  $('modal-add-btn').addEventListener('click', async () => {
    const count = Math.max(1, parseInt($('modal-count').value, 10) || 1);
    const zone  = $('modal-zone').value;
    await addCard(cardData.name, count, zone);
    closeCardModal();
  });

  $('modal-count').addEventListener('keydown', e => {
    if (e.key === 'Enter') $('modal-add-btn').click();
  });

  modalOverlay.classList.add('open');
}

export function closeCardModal() {
  modalOverlay.classList.remove('open');
}

$('modal-close').addEventListener('click', closeCardModal);
modalOverlay.addEventListener('click', e => {
  if (e.target === modalOverlay) closeCardModal();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && modalOverlay.classList.contains('open')) closeCardModal();
});
