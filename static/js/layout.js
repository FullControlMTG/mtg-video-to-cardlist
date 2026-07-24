// Edge-based panel docking with resize handles.
//
// Layout state:
//   {
//     columns:     [['panel-a', 'panel-b'], ['panel-c']],   // ordered
//     columnSizes: [1, 1],                                   // flex-grow per column
//     panelSizes:  { 'panel-a': 1, 'panel-b': 1, 'panel-c': 1 }, // flex-grow per panel
//   }
//
// The DOM mirrors the model: .layout is a flex row of .layout-column
// elements, each a flex column of .panel elements, with .resize-handle
// nodes interleaved between adjacent siblings on both axes. Panels are
// reparented (not re-rendered) so the video stream, deck rows, and
// other stateful child content survive drops.
//
// Dropping on a panel snaps to the nearest edge and inserts the
// dragged panel there:
//   left/right  → new column on that side of the target's column
//   top/bottom  → adjacent row inside the target's column
//
// Dragging a resize handle adjusts the flex-grow of the two adjacent
// elements proportionally so the total size of the pair is preserved.
//
// Layout persists to localStorage.

const STORAGE_KEY = 'layout.state';
const PANEL_IDS = ['panel-video', 'panel-detected', 'panel-deck'];
const MIN_SIZE = 80;   // px — minimum size any panel or column can be resized to

const DEFAULT_STATE = () => ({
  columns:     [['panel-video', 'panel-detected'], ['panel-deck']],
  columnSizes: [1, 1],
  panelSizes:  { 'panel-video': 1, 'panel-detected': 1, 'panel-deck': 1 },
});

const layoutEl = document.getElementById('layout');
let state = load() || DEFAULT_STATE();
let draggedId = null;

// ── Persistence ────────────────────────────────────────────────────
function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state)); } catch { /* private mode */ }
}

function load() {
  let raw;
  try { raw = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null'); } catch { return null; }
  if (!raw || typeof raw !== 'object') return null;
  if (!Array.isArray(raw.columns) || !raw.columns.every(c => Array.isArray(c))) return null;
  const flat = raw.columns.flat();
  const wanted = new Set(PANEL_IDS);
  if (flat.length !== wanted.size || !flat.every(id => wanted.has(id))) return null;
  if (new Set(flat).size !== flat.length) return null;
  if (!Array.isArray(raw.columnSizes) || raw.columnSizes.length !== raw.columns.length) return null;
  if (!raw.panelSizes || typeof raw.panelSizes !== 'object') return null;
  return raw;
}

function avg(nums) { return nums.reduce((a, b) => a + b, 0) / nums.length; }

// ── Rendering ──────────────────────────────────────────────────────
function render() {
  // Order matters: build + attach new columns FIRST (moving panels into
  // them via appendChild), THEN remove the old ones. Doing it the other
  // way round detaches the panels from the document before we can
  // reparent them — getElementById would then return null and .layout
  // would render empty.
  const oldColumns = Array.from(
    layoutEl.querySelectorAll(':scope > .layout-column, :scope > .resize-handle')
  );

  const cols = state.columns.map((colIds, colIdx) => {
    const col = document.createElement('div');
    col.className = 'layout-column';
    col.style.flex = `${state.columnSizes[colIdx]} 1 0`;
    colIds.forEach(id => {
      const panel = document.getElementById(id);
      if (!panel) return;
      panel.style.flex = `${state.panelSizes[id] ?? 1} 1 0`;
      col.appendChild(panel);
    });
    return col;
  });

  // Row handles between adjacent panels inside each column. We query the
  // col's own direct children instead of document.getElementById because
  // at this point the cols are still detached from the document — the
  // panels are already inside them, but the cols haven't been appended
  // to .layout yet, so getElementById would return null.
  cols.forEach(col => {
    const panels = Array.from(col.querySelectorAll(':scope > .panel'));
    for (let r = 0; r < panels.length - 1; r++) {
      col.insertBefore(makeHandle('row', panels[r], panels[r + 1]), panels[r + 1]);
    }
  });

  // Attach columns with column handles between them.
  cols.forEach((col, i) => {
    if (i > 0) layoutEl.appendChild(makeHandle('col', cols[i - 1], col));
    layoutEl.appendChild(col);
  });

  oldColumns.forEach(el => el.remove());
}

// ── Model mutation ─────────────────────────────────────────────────
function findLocation(id) {
  for (let c = 0; c < state.columns.length; c++) {
    const r = state.columns[c].indexOf(id);
    if (r !== -1) return { c, r };
  }
  return null;
}

function removePanel(id) {
  for (let c = 0; c < state.columns.length; c++) {
    const r = state.columns[c].indexOf(id);
    if (r === -1) continue;
    state.columns[c].splice(r, 1);
    if (state.columns[c].length === 0) {
      state.columns.splice(c, 1);
      state.columnSizes.splice(c, 1);
    }
    return;
  }
}

function dropRelative(draggedId, targetId, side) {
  if (draggedId === targetId) return;
  removePanel(draggedId);

  const loc = findLocation(targetId);
  if (!loc) return;
  const { c, r } = loc;

  if (side === 'left' || side === 'right') {
    // New column at c or c+1 — inherit the average size of existing columns
    // so it isn't crushed next to already-resized neighbours.
    const insertAt = side === 'left' ? c : c + 1;
    state.columns.splice(insertAt, 0, [draggedId]);
    state.columnSizes.splice(insertAt, 0, avg(state.columnSizes));
  } else if (side === 'top' || side === 'bottom') {
    const insertAt = side === 'top' ? r : r + 1;
    state.columns[c].splice(insertAt, 0, draggedId);
    // Panel keeps whatever size it had, but if it's the first time it lands
    // here, seed with the average of siblings.
    if (state.panelSizes[draggedId] == null) {
      state.panelSizes[draggedId] = avg(state.columns[c].map(id => state.panelSizes[id] ?? 1));
    }
  }

  save();
  render();
}

// ── Edge detection (for docking) ───────────────────────────────────
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

function clearAllIndicators() {
  document.querySelectorAll('.panel[data-drop-side]')
    .forEach(p => delete p.dataset.dropSide);
}

// ── Resize handle ──────────────────────────────────────────────────
function makeHandle(orientation, prev, next) {
  const h = document.createElement('div');
  h.className = `resize-handle resize-handle-${orientation}`;
  h.addEventListener('mousedown', e => {
    e.preventDefault();
    const isCol = orientation === 'col';         // vertical bar → horizontal drag → column widths
    const axis  = isCol ? 'clientX' : 'clientY';
    const start = e[axis];
    const prevPx = isCol ? prev.offsetWidth  : prev.offsetHeight;
    const nextPx = isCol ? next.offsetWidth  : next.offsetHeight;
    const total  = prevPx + nextPx;

    h.classList.add('dragging');
    document.body.style.cursor = isCol ? 'col-resize' : 'row-resize';
    document.body.style.userSelect = 'none';

    function onMove(ev) {
      const delta = ev[axis] - start;
      const newPrev = Math.max(MIN_SIZE, Math.min(total - MIN_SIZE, prevPx + delta));
      const newNext = total - newPrev;
      // flex-grow doubles as our size unit — the actual pixel widths only
      // depend on the ratio between siblings, so raw px numbers work fine.
      prev.style.flex = `${newPrev} 1 0`;
      next.style.flex = `${newNext} 1 0`;
    }
    function onUp() {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
      h.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      persistSizes();
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
  });
  return h;
}

function persistSizes() {
  // Read the flex-grow values back from the DOM into state.
  const cols = layoutEl.querySelectorAll(':scope > .layout-column');
  state.columnSizes = Array.from(cols).map(c => parseFloat(c.style.flex) || 1);
  PANEL_IDS.forEach(id => {
    const p = document.getElementById(id);
    if (p) state.panelSizes[id] = parseFloat(p.style.flex) || 1;
  });
  save();
}

// ── Docking (drag panel-header, drop on panel edge) ────────────────
function wire(panel) {
  const header = panel.querySelector('.panel-header');
  if (!header) return;

  header.addEventListener('dragstart', e => {
    draggedId = panel.id;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', panel.id);   // Firefox requires setData
    panel.classList.add('dragging');
  });

  header.addEventListener('dragend', () => {
    panel.classList.remove('dragging');
    clearAllIndicators();
    draggedId = null;
  });

  panel.addEventListener('dragover', e => {
    if (!draggedId || draggedId === panel.id) return;
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    const side = nearestEdge(e.clientX, e.clientY, panel.getBoundingClientRect());
    clearAllIndicators();
    panel.dataset.dropSide = side;
  });

  panel.addEventListener('dragleave', e => {
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
