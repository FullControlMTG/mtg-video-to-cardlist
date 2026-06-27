// Shared mutable UI state. ES-module exports are live bindings, so every
// importer sees the same object reference — assign onto fields, never
// reassign the export itself.

export const state = {
  detectedCards: new Map(),   // name -> card data (set by detected.js)
  activeZone: 'main',         // toggled by deck zone tabs
  deck: { main: [], side: [], total_main: 0, total_side: 0 },
};
