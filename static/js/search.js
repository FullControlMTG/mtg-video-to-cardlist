// Header search bar + dropdown. Typing queries /api/search; clicking
// a result opens the detail modal, clicking "+ Add" adds it to the
// current zone.

import { $, escHtml, debounce } from './dom.js';
import { state } from './state.js';
import { api } from './api.js';
import { addCard } from './deck.js';
import { openCardModal } from './modal.js';

const searchInput   = $('search-input');
const searchResults = $('search-results');

// Cache of the most recently rendered results so the Enter handler can
// pick the top one even though it lives outside renderSearchResults' scope.
let _currentResults = [];

function resetSearch() {
  _currentResults = [];
  searchInput.value = '';
  searchResults.classList.remove('open');
  searchResults.innerHTML = '';
}

const doSearch = debounce(async () => {
  const q = searchInput.value.trim();
  if (q.length < 2) {
    _currentResults = [];
    searchResults.classList.remove('open');
    searchResults.innerHTML = '';
    return;
  }
  const data = await api.searchCards(q);
  renderSearchResults(data?.results || []);
}, 280);

searchInput.addEventListener('input', doSearch);

searchInput.addEventListener('blur', () => {
  // Delay so clicks on dropdown items register before the dropdown closes.
  setTimeout(() => searchResults.classList.remove('open'), 200);
});

searchInput.addEventListener('focus', () => {
  if (searchResults.children.length) searchResults.classList.add('open');
});

searchInput.addEventListener('keydown', async e => {
  if (e.key === 'Escape') {
    searchResults.classList.remove('open');
    searchInput.blur();
    return;
  }
  if (e.key === 'Enter') {
    e.preventDefault();
    // Add the top result to the active zone, then wipe the input + dropdown
    // and re-focus so the user can keep typing the next card name.
    if (!_currentResults.length) return;
    const top = _currentResults[0];
    resetSearch();
    await addCard(top.name, 1, state.activeZone);
    searchInput.focus();
  }
});

function renderSearchResults(results) {
  _currentResults = results;
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
      const name = btn.dataset.name;
      resetSearch();
      await addCard(name, 1, state.activeZone);
      searchInput.focus();
    });
  });
}
