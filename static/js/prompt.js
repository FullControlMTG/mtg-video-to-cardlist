// In-app replacement for `window.prompt()` — uses the app's existing modal
// styling so the popup matches the dark theme. Build the DOM on demand and
// remove it on close; no scaffolding needed in index.html.

import { escHtml } from './dom.js';

/**
 * Ask the user for a number.
 * @returns {Promise<number|null>} the parsed value, or null if cancelled.
 */
export function promptNumber({ title, label, value = 1, min = 0, max = 99 }) {
  return new Promise(resolve => {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay open';
    overlay.innerHTML = `
      <div class="modal modal-prompt" role="dialog" aria-modal="true" aria-label="${escHtml(title)}">
        <div class="prompt-title">${escHtml(title)}</div>
        <label class="prompt-field">
          <span>${escHtml(label)}</span>
          <input type="number" class="prompt-input"
                 value="${value}" min="${min}" max="${max}" inputmode="numeric" />
        </label>
        <div class="prompt-actions">
          <button class="btn-ghost prompt-cancel">Cancel</button>
          <button class="btn-primary prompt-save">Save</button>
        </div>
      </div>
    `;
    document.body.appendChild(overlay);

    const input  = overlay.querySelector('.prompt-input');
    const save   = overlay.querySelector('.prompt-save');
    const cancel = overlay.querySelector('.prompt-cancel');

    const close = result => {
      document.removeEventListener('keydown', onKey);
      overlay.remove();
      resolve(result);
    };

    const submit = () => {
      const raw = input.value.trim();
      if (raw === '') return close(null);
      const n = parseInt(raw, 10);
      if (!Number.isFinite(n)) return close(null);
      close(Math.max(min, Math.min(max, n)));
    };

    const onKey = e => {
      if (e.key === 'Escape') close(null);
      else if (e.key === 'Enter' && document.activeElement === input) submit();
    };

    save.addEventListener('click', submit);
    cancel.addEventListener('click', () => close(null));
    overlay.addEventListener('click', e => { if (e.target === overlay) close(null); });
    document.addEventListener('keydown', onKey);

    // Focus + select so the user can immediately overwrite the value.
    input.focus();
    input.select();
  });
}
