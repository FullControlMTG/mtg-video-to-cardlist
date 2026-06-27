// Deck settings modal (gear icon in the sidebar header). Edits the
// deck-level metadata: name, format, commander, notes.

import { $ } from './dom.js';
import { api } from './api.js';

const overlay        = $('settings-overlay');
const formatSelect   = $('settings-format');
const commanderRow   = $('settings-commander-row');
const deckTitle      = $('deck-name-display');

function toggleCmdrRow() {
  const fmt = (formatSelect.value || '').toLowerCase();
  const needsCmdr = ['commander', 'brawl', 'historic brawl', 'oathbreaker'].includes(fmt);
  commanderRow.style.display = needsCmdr ? '' : 'none';
}

async function openSettings() {
  const meta = await api.getDeckMeta();
  if (!meta) return;

  $('settings-name').value      = meta.name      || '';
  $('settings-format').value    = meta.format    || '';
  $('settings-commander').value = meta.commander || '';
  $('settings-notes').value     = meta.notes     || '';

  // Populate the format dropdown once, from the server's authoritative list.
  if (formatSelect.options.length <= 1) {
    (meta.formats || []).forEach(f => {
      const opt = document.createElement('option');
      opt.value = opt.textContent = f;
      formatSelect.appendChild(opt);
    });
  }

  toggleCmdrRow();
  overlay.classList.add('open');
}

formatSelect.addEventListener('change', toggleCmdrRow);

$('settings-save').addEventListener('click', async () => {
  const body = {
    name:      $('settings-name').value.trim(),
    format:    $('settings-format').value,
    commander: $('settings-commander').value.trim(),
    notes:     $('settings-notes').value.trim(),
  };
  await api.patchDeckMeta(body);
  overlay.classList.remove('open');
  if (deckTitle) deckTitle.textContent = body.name || 'Decklist';
});

$('settings-close').addEventListener('click', () => overlay.classList.remove('open'));
overlay.addEventListener('click', e => {
  if (e.target === overlay) overlay.classList.remove('open');
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && overlay.classList.contains('open')) overlay.classList.remove('open');
});

$('settings-btn').addEventListener('click', openSettings);
