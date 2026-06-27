// Sidebar collapse/expand toggle. Adding `.deck-hidden` to the layout
// element drives all the visual changes via CSS (column width, hidden
// children, visible reopen tab).

import { $ } from './dom.js';

const layoutEl = $('layout');

$('toggle-deck-btn').addEventListener('click', () => layoutEl.classList.add('deck-hidden'));
$('show-deck-btn').addEventListener('click',   () => layoutEl.classList.remove('deck-hidden'));
