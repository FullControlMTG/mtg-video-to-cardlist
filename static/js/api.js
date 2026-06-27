// Thin wrappers around the FastAPI endpoints. Every UI module calls into
// these instead of using fetch() directly, so request shape changes happen
// in one place.

async function jsonOk(promise) {
  const resp = await promise;
  if (!resp.ok) return null;
  return resp.json();
}

export const api = {
  // Deck
  getDeck:        () => jsonOk(fetch('/api/cards')),
  addCard:        (name, count, zone) => fetch('/api/cards', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, count, zone }),
                  }),
  setCount:       (name, count, zone) => fetch(`/api/cards/${encodeURIComponent(name)}`, {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ count, zone }),
                  }),
  decrementOne:   (name, zone) => fetch(`/api/cards/${encodeURIComponent(name)}?count=1&zone=${zone}`, {
                    method: 'DELETE',
                  }),
  clearDeck:      () => fetch('/api/deck/clear', { method: 'POST' }),

  // Deck metadata
  getDeckMeta:    () => jsonOk(fetch('/api/deck')),
  patchDeckMeta:  body => fetch('/api/deck', {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(body),
                  }),

  // Cards / search
  searchCards:    q => jsonOk(fetch(`/api/search?q=${encodeURIComponent(q)}`)),
  getCard:        name => jsonOk(fetch(`/api/card/${encodeURIComponent(name)}`)),

  // Export
  exportDeck:     async fmt => {
                    const resp = await fetch(`/api/export/${fmt}`);
                    return resp.ok ? resp.text() : null;
                  },

  // Camera
  getCameras:     () => jsonOk(fetch('/api/cameras')),
  getCameraStatus:() => jsonOk(fetch('/api/camera/status')),
  selectCamera:   source => fetch('/api/cameras/select', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ source }),
                  }),
  rotateCamera:   rotation => fetch('/api/camera/rotate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ rotation }),
                  }),
};
