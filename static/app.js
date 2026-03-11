'use strict';

const state = {
  detectedCards: new Map(),
  activeZone: 'main',
  deck: { main: [], side: [], total_main: 0, total_side: 0 },
  ws: null,
  currentModalCard: null,
};

const $ = id => document.getElementById(id);

const wsDot         = $('ws-dot');
const searchInput   = $('search-input');
const searchResults = $('search-results');
const detectedGrid  = $('detected-grid');
const scanBadge     = $('scan-badge');
const deckMain      = $('decklist-main');
const deckSide      = $('decklist-side');
const mainCount     = $('main-count');
const sideCount     = $('side-count');
const modalOverlay  = $('modal-overlay');
const modalBody     = $('modal-body');
const exportOverlay = $('export-overlay');
const exportTitle   = $('export-title');
const exportText    = $('export-text');
const paneRight     = $('pane-right');
const deckCollapsed = $('deck-collapsed');
const helpOverlay   = $('help-overlay');

const videoFeed = $('video-feed');

function reloadVideoFeed() {
  videoFeed.src = '';
  setTimeout(() => { videoFeed.src = '/video'; }, 400);
}

videoFeed.addEventListener('error', () => {
  setTimeout(reloadVideoFeed, 2000);
});

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function escHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    wsDot.classList.add('connected');
    wsDot.classList.remove('error');
    wsDot.title = 'WebSocket connected';
    state.ws = ws;
  };

  ws.onclose = () => {
    wsDot.classList.remove('connected');
    wsDot.classList.remove('error');
    wsDot.title = 'Disconnected – reconnecting…';
    state.ws = null;
    setTimeout(connectWS, 2500);
  };

  ws.onerror = () => {
    wsDot.classList.add('error');
    wsDot.classList.remove('connected');
    wsDot.title = 'WebSocket error';
  };

  ws.onmessage = e => {
    let msg;
    try { msg = JSON.parse(e.data); } catch { return; }
    if (msg.type === 'detected') handleDetected(msg.cards || []);
    else if (msg.type === 'deck_update') applyDeckUpdate(msg);
  };
}

async function handleDetected(cards) {
  for (const c of cards) {
    if (!c.name || state.detectedCards.has(c.name)) continue;

    state.detectedCards.set(c.name, { name: c.name });

    let cardData = null;
    try {
      const resp = await fetch(`/api/card/${encodeURIComponent(c.name)}`);
      if (resp.ok) cardData = await resp.json();
    } catch { /* ignore */ }

    const data = cardData || { name: c.name };
    state.detectedCards.set(c.name, data);
    renderDetectedCard(c.name, data);
  }

  const n = state.detectedCards.size;
  scanBadge.textContent = n ? `${n} found` : 'scanning…';
}

function renderDetectedCard(name, cardData) {
  const hint = detectedGrid.querySelector('.empty-hint');
  if (hint) hint.remove();

  const img = cardData?.image_uri || '';
  const div = document.createElement('div');
  div.className = 'detected-card';
  div.dataset.name = name;
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
    if (!detectedGrid.querySelector('.detected-card')) {
      detectedGrid.innerHTML = '<p class="empty-hint">Cards recognised by the camera will appear here.</p>';
    }
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

  detectedGrid.appendChild(div);
}

$('clear-detected-btn').addEventListener('click', () => {
  state.detectedCards.clear();
  detectedGrid.innerHTML = '<p class="empty-hint">Cards recognised by the camera will appear here.</p>';
  scanBadge.textContent = 'scanning…';
});

async function loadDeck() {
  try {
    const resp = await fetch('/api/cards');
    if (!resp.ok) return;
    applyDeckUpdate(await resp.json());
  } catch { /* ignore */ }
}

function applyDeckUpdate(data) {
  state.deck = data;
  renderZone('main', data.main || []);
  renderZone('side', data.side || []);
  mainCount.textContent = data.total_main || 0;
  sideCount.textContent = data.total_side || 0;
}

function renderZone(zone, entries) {
  const el = zone === 'main' ? deckMain : deckSide;

  if (!entries.length) {
    el.innerHTML = '<div class="deck-empty">No cards yet.</div>';
    return;
  }

  el.innerHTML = entries.map(e => `
    <div class="deck-row" data-name="${escHtml(e.name)}" data-zone="${zone}">
      <img class="deck-row-thumb"
           src="${escHtml(e.image_uri || '')}"
           alt="${escHtml(e.name)}"
           onerror="this.style.visibility='hidden'" />
      <div>
        <div class="deck-row-name">${escHtml(e.name)}</div>
        <div class="deck-row-mana">${escHtml(e.mana_cost || '')}</div>
      </div>
      <div class="deck-row-count">${e.count}</div>
      <button class="deck-row-remove"
              title="Remove one"
              data-name="${escHtml(e.name)}"
              data-zone="${zone}">×</button>
    </div>
  `).join('');

  el.querySelectorAll('.deck-row').forEach(row => {
    row.addEventListener('click', e => {
      if (e.target.classList.contains('deck-row-remove')) return;
      const entry = entries.find(x => x.name === row.dataset.name);
      if (entry) openCardModal(entry);
    });
  });

  el.querySelectorAll('.deck-row-remove').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.stopPropagation();
      const { name, zone: z } = btn.dataset;
      await fetch(`/api/cards/${encodeURIComponent(name)}?zone=${z}`, { method: 'DELETE' });
      await loadDeck();
    });
  });
}

document.querySelectorAll('.zone-tab').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.zone-tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    state.activeZone = tab.dataset.zone;
    deckMain.style.display = state.activeZone === 'main' ? '' : 'none';
    deckSide.style.display = state.activeZone === 'side' ? '' : 'none';
  });
});

const _doSearch = debounce(async () => {
  const q = searchInput.value.trim();
  if (q.length < 2) {
    searchResults.classList.remove('open');
    searchResults.innerHTML = '';
    return;
  }
  try {
    const resp = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    if (!resp.ok) return;
    const data = await resp.json();
    renderSearchResults(data.results || []);
  } catch { /* ignore */ }
}, 280);

searchInput.addEventListener('input', _doSearch);

searchInput.addEventListener('blur', () => {
  // Delay so clicks on dropdown items register before the dropdown closes
  setTimeout(() => searchResults.classList.remove('open'), 200);
});

searchInput.addEventListener('focus', () => {
  if (searchResults.children.length) searchResults.classList.add('open');
});

searchInput.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    searchResults.classList.remove('open');
    searchInput.blur();
  }
});

function renderSearchResults(results) {
  if (!results.length) {
    searchResults.classList.remove('open');
    searchResults.innerHTML = '';
    return;
  }

  const byName = Object.fromEntries(results.map(c => [c.name, c]));

  searchResults.innerHTML = results.map(c => `
    <div class="search-result-item" data-name="${escHtml(c.name)}" tabindex="0">
      <img src="${escHtml(c.image_uri || '')}" alt="${escHtml(c.name)}"
           onerror="this.style.visibility='hidden'" />
      <div>
        <div class="search-result-name">${escHtml(c.name)}</div>
        <div class="search-result-type">${escHtml(c.type_line || '')}</div>
      </div>
      <button class="search-add-btn" data-name="${escHtml(c.name)}">+ Add</button>
    </div>
  `).join('');

  searchResults.classList.add('open');

  searchResults.querySelectorAll('.search-result-item').forEach(item => {
    item.addEventListener('click', e => {
      if (e.target.classList.contains('search-add-btn')) return;
      searchResults.classList.remove('open');
      openCardModal(byName[item.dataset.name]);
    });
  });

  searchResults.querySelectorAll('.search-add-btn').forEach(btn => {
    btn.addEventListener('click', async e => {
      e.stopPropagation();
      await addCard(btn.dataset.name, 1, state.activeZone);
      searchResults.classList.remove('open');
      searchInput.value = '';
    });
  });
}

function openCardModal(cardData) {
  state.currentModalCard = cardData;

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
    closeModal();
  });

  $('modal-count').addEventListener('keydown', e => {
    if (e.key === 'Enter') $('modal-add-btn').click();
  });

  modalOverlay.classList.add('open');
}

function closeModal() {
  modalOverlay.classList.remove('open');
  state.currentModalCard = null;
}

$('modal-close').addEventListener('click', closeModal);
modalOverlay.addEventListener('click', e => {
  if (e.target === modalOverlay) closeModal();
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    if (modalOverlay.classList.contains('open'))  closeModal();
    if (exportOverlay.classList.contains('open')) exportOverlay.classList.remove('open');
  }
});

async function addCard(name, count = 1, zone = 'main') {
  try {
    const resp = await fetch('/api/cards', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, count, zone }),
    });
    if (!resp.ok) {
      console.error('Add card error:', await resp.text());
      return;
    }
    // Also refresh directly in case the WS deck_update is delayed
    await loadDeck();
  } catch (err) {
    console.error('Add card failed:', err);
  }
}

document.querySelectorAll('.btn-export').forEach(btn => {
  btn.addEventListener('click', async () => {
    const fmt = btn.dataset.fmt;
    try {
      const resp = await fetch(`/api/export/${fmt}`);
      if (!resp.ok) return;
      const text = await resp.text();
      exportTitle.textContent = `Export – ${fmt.toUpperCase()}`;
      exportText.value = text;
      exportOverlay.classList.add('open');
    } catch { /* ignore */ }
  });
});

$('export-close').addEventListener('click', () => exportOverlay.classList.remove('open'));
exportOverlay.addEventListener('click', e => {
  if (e.target === exportOverlay) exportOverlay.classList.remove('open');
});

$('copy-btn').addEventListener('click', () => {
  navigator.clipboard.writeText(exportText.value).then(() => {
    const btn = $('copy-btn');
    btn.textContent = 'Copied!';
    setTimeout(() => { btn.textContent = 'Copy to Clipboard'; }, 1800);
  });
});

$('download-btn').addEventListener('click', () => {
  const blob = new Blob([exportText.value], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'decklist.txt';
  a.click();
  URL.revokeObjectURL(a.href);
});

$('clear-deck-btn').addEventListener('click', async () => {
  if (!confirm('Clear the entire decklist? This cannot be undone.')) return;
  await fetch('/api/deck/clear', { method: 'POST' });
  await loadDeck();
});

const cameraSelect = $('camera-select');

cameraSelect.addEventListener('change', async () => {
  const source = parseInt(cameraSelect.value, 10);
  if (isNaN(source)) return;
  try {
    await fetch('/api/cameras/select', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source }),
    });
    reloadVideoFeed();
  } catch { /* ignore */ }
});

const settingsOverlay = $('settings-overlay');
const settingsFormat  = $('settings-format');
const settingsCmdrRow = $('settings-commander-row');

async function openSettings() {
  try {
    const resp = await fetch('/api/deck');
    if (!resp.ok) return;
    const meta = await resp.json();

    $('settings-name').value      = meta.name      || '';
    $('settings-format').value    = meta.format    || '';
    $('settings-commander').value = meta.commander || '';
    $('settings-notes').value     = meta.notes     || '';

    if (settingsFormat.options.length <= 1) {
      (meta.formats || []).forEach(f => {
        const opt = document.createElement('option');
        opt.value = opt.textContent = f;
        settingsFormat.appendChild(opt);
      });
    }

    toggleCmdrRow();
    settingsOverlay.classList.add('open');
  } catch { /* ignore */ }
}

function toggleCmdrRow() {
  const fmt = (settingsFormat.value || '').toLowerCase();
  const needsCmdr = ['commander','brawl','historic brawl','oathbreaker'].includes(fmt);
  settingsCmdrRow.style.display = needsCmdr ? '' : 'none';
}

settingsFormat.addEventListener('change', toggleCmdrRow);

$('settings-save').addEventListener('click', async () => {
  const body = {
    name:      $('settings-name').value.trim(),
    format:    $('settings-format').value,
    commander: $('settings-commander').value.trim(),
    notes:     $('settings-notes').value.trim(),
  };
  try {
    await fetch('/api/deck', {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    settingsOverlay.classList.remove('open');
    updateDeckTitle(body.name);
  } catch { /* ignore */ }
});

$('settings-close').addEventListener('click', () => settingsOverlay.classList.remove('open'));
settingsOverlay.addEventListener('click', e => {
  if (e.target === settingsOverlay) settingsOverlay.classList.remove('open');
});

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && settingsOverlay.classList.contains('open'))
    settingsOverlay.classList.remove('open');
});

$('settings-btn').addEventListener('click', openSettings);

function updateDeckTitle(name) {
  const el = $('deck-name-display');
  if (el) el.textContent = name || 'Decklist';
}

$('toggle-deck-btn').addEventListener('click', () => {
  paneRight.style.display = 'none';
  deckCollapsed.style.display = 'flex';
});

$('show-deck-btn').addEventListener('click', () => {
  paneRight.style.display = '';
  deckCollapsed.style.display = 'none';
});

$('help-btn').addEventListener('click', () => helpOverlay.classList.add('open'));
$('help-close').addEventListener('click', () => helpOverlay.classList.remove('open'));
helpOverlay.addEventListener('click', e => {
  if (e.target === helpOverlay) helpOverlay.classList.remove('open');
});
document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && helpOverlay.classList.contains('open'))
    helpOverlay.classList.remove('open');
});

connectWS();
loadDeck();

async function loadCameras() {
  try {
    const resp = await fetch('/api/cameras');
    if (!resp.ok) return;
    const { cameras, current } = await resp.json();

    cameraSelect.innerHTML = cameras.length
      ? cameras.map(c =>
          `<option value="${c.index}">${c.name} (${c.resolution})</option>`
        ).join('')
      : '<option value="">No cameras found</option>';

    cameraSelect.value = String(current);
  } catch { /* ignore */ }
}

loadCameras();
