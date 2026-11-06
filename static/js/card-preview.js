// Large card image shown beside the pointer's card while hovering a decklist
// row or a detected card.

const GAP = 12;
const MARGIN = 8;
const HOVER_DELAY_MS = 600;

const preview = document.createElement('img');
preview.className = 'card-preview';
preview.alt = '';
preview.hidden = true;
preview.addEventListener('error', hideCardPreview);
document.body.appendChild(preview);

let hovered = null;
let showTimer = null;

export function hideCardPreview() {
  clearTimeout(showTimer);
  hovered = null;
  preview.hidden = true;
}

function showCardPreview(card, src) {
  if (preview.getAttribute('src') !== src) preview.src = src;
  preview.hidden = false;

  const rect = card.getBoundingClientRect();
  const { offsetWidth: w, offsetHeight: h } = preview;
  const roomLeft = rect.left - GAP - MARGIN;
  const roomRight = innerWidth - rect.right - GAP - MARGIN;
  const left = roomRight >= w || roomRight >= roomLeft
    ? rect.right + GAP
    : rect.left - GAP - w;
  const top = rect.top + rect.height / 2 - h / 2;

  preview.style.left = `${Math.max(MARGIN, Math.min(left, innerWidth - w - MARGIN))}px`;
  preview.style.top = `${Math.max(MARGIN, Math.min(top, innerHeight - h - MARGIN))}px`;
}

document.addEventListener('mouseover', e => {
  const card = e.target.closest?.('.deck-row, .detected-card');
  if (card === hovered) return;
  // e.buttons: a panel resize or text drag is passing over the cards.
  const src = card && !e.buttons ? card.querySelector('img')?.getAttribute('src') : '';
  hideCardPreview();
  if (!src) return;
  hovered = card;
  showTimer = setTimeout(() => showCardPreview(card, src), HOVER_DELAY_MS);
});

document.documentElement.addEventListener('mouseleave', hideCardPreview);
document.addEventListener('click', hideCardPreview, true);
// Capture, because scroll events from the inner lists don't bubble.
document.addEventListener('scroll', hideCardPreview, true);
