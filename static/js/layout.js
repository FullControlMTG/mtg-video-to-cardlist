// Edge-based panel docking.
//
// Layout is stored as an array of columns; each column is an ordered
// list of panel IDs. The DOM mirrors this: .layout is a flex row of
// .layout-column elements, each a flex column of .panel elements.
// Rendering reparents the actual .panel nodes into the correct column
// (moving DOM nodes, not innerHTML rewriting), so event listeners and
// child state — the video stream, deck rows, etc. — are preserved.
//
// Dropping on a panel snaps to the nearest edge and inserts the
// dragged panel there:
//   left/right  → new column on that side of the target's column
//   top/bottom  → adjacent row inside the target's column
//
// Layout persists to localStorage.

const STORAGE_KEY = 'layout.columns';
const PANEL_IDS = ['panel-video', 'panel-detected', 'panel-deck'];
const DEFAULT_LAYOUT = [
  ['panel-video', 'panel-detected'],
  ['panel-deck'],
];

const layoutEl = document.getElementById('layout');
let columns = load() || DEFAULT_LAYOUT;
let draggedId = null;

// ── Persistence ────────────────────────────────────────────────────
function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(columns)); } catch { /* private mode */ }
}

function load() {
  let raw;
  try { raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch { return null; }
  // Validate: an array of arrays that together contain each panel exactly once.
  if (!Array.isArray(raw) || !raw.every(c => Array.isArray(c))) return null;
  const flat = raw.flat();
  const wanted = new Set(PANEL_IDS);
  if (flat.length !== wanted.size || !flat.every(id => wanted.has(id))) return null;
  if (new Set(flat).size !== flat.length) return null;
  return raw;
}

// ── Rendering ──────────────────────────────────────────────────────
function render() {
  // Order matters: build + attach new columns FIRST (moving panels into
  // them via appendChild), THEN remove the old ones. Doing it the other
  // way round detaches the panels from the document before we can
  // reparent them — getElementById would then return null and .layout
  // would render empty (grey background bug).
  const oldColumns = Array.from(
    layoutEl.querySelectorAll(':scope > .layout-column')
  );

  columns.forEach(colIds => {
    const col = document.createElement('div');
    col.className = 'layout-column';
    colIds.forEach(id => {
      const panel = document.getElementById(id);
      if (panel) col.appendChild(panel);   // moves; panel stays in document
    });
    layoutEl.appendChild(col);
  });

  oldColumns.forEach(c => c.remove());     // now empty, safe to drop
}

// ── Model mutation ─────────────────────────────────────────────────
function findLocation(id) {
  for (let c = 0; c < columns.length; c++) {
    const r = columns[c].indexOf(id);
    if (r !== -1) return { c, r };
  }
  return null;
}

function removePanel(id) {
  columns = columns
    .map(col => col.filter(p => p !== id))
    .filter(col => col.length > 0);
}

function dropRelative(draggedId, targetId, side) {
  if (draggedId === targetId) return;
  removePanel(draggedId);

  const loc = findLocation(targetId);
  if (!loc) return;
  const { c, r } = loc;

  if      (side === 'left')   columns.splice(c,     0, [draggedId]);
  else if (side === 'right')  columns.splice(c + 1, 0, [draggedId]);
  else if (side === 'top')    columns[c].splice(r,     0, draggedId);
  else if (side === 'bottom') columns[c].splice(r + 1, 0, draggedId);

  save();
  render();
}

// ── Edge detection ─────────────────────────────────────────────────
function nearestEdge(x, y, rect) {
  const dt = y - rect.top;
  const db = rect.bottom - y;
  const dl = x - rect.left;
  const dr = rect.right - x;
  const m = Math.min(dt, db, dl, dr);
  if (m === dt) return 'top';
  if (m === db) return 'bottom';
  if (m === dl) return 'left';
  return 'right';
}

function setDropIndicator(panel, side) {
  panel.dataset.dropSide = side;
}
function clearAllIndicators() {
  document.querySelectorAll('.panel[data-drop-side]')
    .forEach(p => delete p.dataset.dropSide);
}

// ── Wiring per panel ───────────────────────────────────────────────
function wire(panel) {
  const header = panel.querySelector('.panel-header');
  if (!header) return;

  header.addEventListener('dragstart', e => {
    draggedId = panel.id;
    e.dataTransfer.effectAllowed = 'move';
    // Firefox requires setData or the drag doesn't start.
    e.dataTransfer.setData('text/plain', panel.id);
    panel.classList.add('dragging');
  });

  header.addEventListener('dragend', () => {
    panel.classList.remove('dragging');
    clearAllIndicators();
    draggedId = null;
  });

  panel.addEventListener('dragover', e => {
    if (!draggedId || draggedId === panel.id) return;
    e.preventDefault();                          // required to allow drop
    e.dataTransfer.dropEffect = 'move';
    const side = nearestEdge(e.clientX, e.clientY, panel.getBoundingClientRect());
    clearAllIndicators();
    setDropIndicator(panel, side);
  });

  panel.addEventListener('dragleave', e => {
    // Only clear when the cursor truly leaves the panel (dragleave fires
    // on transitions into child elements too).
    if (!panel.contains(e.relatedTarget)) delete panel.dataset.dropSide;
  });

  panel.addEventListener('drop', e => {
    e.preventDefault();
    const side = panel.dataset.dropSide;
    clearAllIndicators();
    if (!draggedId || !side) return;
    dropRelative(draggedId, panel.id, side);
  });
}

// ── Boot ───────────────────────────────────────────────────────────
PANEL_IDS.forEach(id => {
  const el = document.getElementById(id);
  if (el) wire(el);
});
render();
