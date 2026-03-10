# MTG Card Scanner

Point a webcam at your Magic: The Gathering cards and watch them build your decklist in real time.

---

## What it does

- **Live card recognition** — hold a card up to the camera; the scanner detects the card shape, runs OCR on the name strip, and fuzzy-matches it against the entire Scryfall database (~30 k cards).
- **Scryfall overlay** — the matched card's artwork and details appear instantly alongside the video feed.
- **Decklist builder** — click any detected card (or search manually) to add it to your main deck or sideboard with a count you choose.
- **Multiple export formats** — copy or download your list in Moxfield, MTGA/Arena, MTGO, or plain-text format.
- **Persistent state** — your decklist survives restarts; Scryfall card data is cached locally and refreshed every 3 days.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Web framework | **FastAPI** + Uvicorn | Async, fast, WebSocket support built-in |
| Video | **OpenCV** | Contour detection + perspective warp for card extraction |
| OCR | **EasyOCR** | Accurate printed-text recognition, no external API needed |
| Name matching | **RapidFuzz** | Tolerant of OCR noise; sub-millisecond fuzzy matching |
| Card data | **Scryfall bulk API** | Free, comprehensive, offline-capable after first download |
| Frontend | Vanilla JS + WebSocket | Zero-dependency; real-time deck updates without polling |

---

## Setup

**Requirements:** Python 3.10+, a webcam, ~500 MB disk (models + Scryfall cache).

```bash
# 1. Clone
git clone <repo-url>
cd mtg-video-to-cardlist

# 2. Create environment (uv recommended)
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# 3. Run
python main.py
```

The browser opens automatically at `http://127.0.0.1:8000`.
On first run, Scryfall bulk data downloads in the background — the UI is usable immediately while it loads.

---

## How the detection pipeline works

```
Webcam frame
  └─ Canny edge detection + contour finding
       └─ Aspect-ratio filter (portrait rectangle ≈ 0.55–0.80 w/h)
            └─ Perspective warp → upright card crop
                 └─ EasyOCR on top 14% of card (name strip)
                      └─ RapidFuzz WRatio match against Scryfall name list
                           └─ WebSocket event → UI overlay + detected grid
```

The scanner runs in a background thread, processing every 5th frame to keep the stream smooth. Duplicate detections are suppressed for 4 seconds.

---

## Configuration

All tunable parameters live in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | `0` | Webcam index or RTSP URL string |
| `FRAME_SKIP` | `5` | Process every Nth frame |
| `MIN_CARD_AREA` | `8000 px²` | Minimum contour area to consider |
| `FUZZY_MATCH_THRESHOLD` | `72` | RapidFuzz score cutoff (0–100) |
| `OCR_MIN_CONFIDENCE` | `0.25` | Minimum EasyOCR confidence to keep a result |
| `PORT` | `8000` | Server port |

---

## Export formats

| Button | Format | Accepted by |
|---|---|---|
| **Moxfield** | Plain `N CardName` | Moxfield, Archidekt, Deckstats |
| **MTGA** | `N CardName (SET) #` | MTG Arena import |
| **MTGO** | Plain `N CardName` | Magic Online `.dek` |
| **Plain Text** | Plain `N CardName` | Universal |

---

## Tips for best recognition

- Use good, even lighting — avoid harsh shadows across the card face.
- Hold the card steady for 1–2 seconds; the scanner debounces detections.
- Cards in sleeves work fine; foils can cause glare — tilt slightly.
- Camera distance of ~20–40 cm from the card works best.
