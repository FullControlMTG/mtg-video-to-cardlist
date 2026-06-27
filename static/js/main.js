// App entry point. Each imported module is responsible for wiring its
// own DOM listeners at top level; this file only kicks off the async
// startup work (WS connect, initial data fetch, status polling).

import './deck.js';        // renderZone, addCard, zone tabs, clear button
import './modal.js';       // card detail modal handlers
import './search.js';      // search input + dropdown
import './detected.js';    // detected-cards panel + clear button
import './export.js';      // export modal + format buttons
import './settings.js';    // deck settings modal
import './help.js';        // help modal
import './layout.js';      // sidebar collapse toggle
import './camera.js';      // (also wires its listeners at import)

import { connectWS } from './ws.js';
import { loadDeck } from './deck.js';
import { loadCameras, pollCameraStatus } from './camera.js';

connectWS();
loadDeck();
loadCameras();
pollCameraStatus();
setInterval(pollCameraStatus, 2000);
