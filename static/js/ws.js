// WebSocket connection to /ws. Dispatches `detected` (new scanned cards)
// and `deck_update` (server-side deck mutation) to the relevant modules.

import { $ } from './dom.js';
import { handleDetected } from './detected.js';
import { applyDeckUpdate } from './deck.js';

const wsDot = $('ws-dot');

export function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    wsDot.classList.add('connected');
    wsDot.classList.remove('error');
    wsDot.title = 'WebSocket connected';
  };

  ws.onclose = () => {
    wsDot.classList.remove('connected');
    wsDot.classList.remove('error');
    wsDot.title = 'Disconnected – reconnecting…';
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
    if      (msg.type === 'detected')    handleDetected(msg.cards || []);
    else if (msg.type === 'deck_update') applyDeckUpdate(msg);
  };
}
