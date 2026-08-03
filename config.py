from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CARD_NAMES_FILE = DATA_DIR / "card_names.json"
# Cards discovered at runtime via the live Scryfall fallback (missing from
# whatever bulk snapshot is in card_names.json). Loaded alongside the bulk
# and grown one card at a time — survives restarts, ages out naturally as
# the bulk gets refreshed.
CARD_NAMES_GROWN_FILE = DATA_DIR / "card_names_grown.json"
DECKLIST_FILE = DATA_DIR / "decklist.json"

# Scryfall
SCRYFALL_BULK_DATA_URL = "https://api.scryfall.com/bulk-data/oracle-cards"
SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/named"
SCRYFALL_AUTOCOMPLETE_URL = "https://api.scryfall.com/cards/autocomplete"

# Video capture
VIDEO_SOURCE = 0          # Webcam index (0 = default). Can be a RTSP URL string.
JPEG_QUALITY = 75         # MJPEG stream quality
CAPTURE_WIDTH = 1280      # Requested capture width (camera may override)
CAPTURE_HEIGHT = 720      # Requested capture height

# Camera lifecycle (threaded grabber + state machine)
CAMERA_WARMUP_TIMEOUT = 6.0      # Seconds to wait for a camera's first frame before giving up
CAMERA_READ_FAIL_LIMIT = 30      # Consecutive failed reads before a reconnect is attempted
CAMERA_RECONNECT_DELAY = 1.0     # Seconds between reconnect attempts
CAMERA_PROBE_MAX = 8             # Highest device index probed when auto-selecting at startup

# Card detection (contour-based)
MIN_CARD_AREA = 8000      # Pixels² — filters out small noise contours
# Reject quads covering more than this fraction of the frame. This kills the
# common false positive where the whole camera image (its border/background) is
# read as one big rectangle; a real card held in frame leaves a margin around it.
MAX_CARD_AREA_FRACTION = 0.92
CARD_ASPECT_MIN = 0.55    # Width/height ratio lower bound (standard card ~0.715)
CARD_ASPECT_MAX = 0.80    # Width/height ratio upper bound
# The detected quad (orange) is well-placed, so the OCR region (red) is just a
# hair larger than it — small safety margins only, NOT a big inflation:
#   CARD_BORDER_MARGIN — uniform outward expansion on all sides.
#   CARD_TOP_EXTRA     — tiny extra UPWARD nudge so the very top of the title
#                        isn't clipped (fraction of card height).
# Tuning dials, watching the yellow OCR box: if the title top is clipped, raise
# CARD_TOP_EXTRA a little; if red looks too tall, lower it (0 = match orange).
CARD_BORDER_MARGIN = 0.03
CARD_TOP_EXTRA = 0.02
# Name strip = top-left region of the (slightly grown) card we OCR — a tight band
# on the title bar.
NAME_ROW_FRACTION = 0.14  # Top fraction (height of the OCR band) — the title bar
NAME_COL_FRACTION = 0.85  # Left fraction (width) — long enough for long card names

# Detection confirmation — vote on the FUZZY-MATCHED CARD NAME, not the raw OCR.
# Every pass: detect rectangle → OCR → fuzzy-match the read to a Scryfall card
# name (which corrects mis-OCR) → record that name. We confirm as soon as the
# same card name wins M of the most recent N reads. Because the fuzzy match
# corrects per-read noise, agreement converges in just a few frames, so this is
# both fast and accurate. No artificial waits.
CONFIRM_WINDOW_SIZE = 8          # N — most-recent frames considered in the vote
CONFIRM_MIN_MATCH = 2            # M — frames that must resolve to the SAME card name to confirm

# Frame-level "is there anything to look at" gate. A camera pointed at the
# black table with the lens covered still produces noise contours that get
# OCR'd into garbage. This threshold is the minimum standard deviation of
# pixel intensities across the greyscale frame — a truly black/uniform
# frame is < 5; a scene with any real content is > 20. Frames below this
# skip detection entirely and record an absent vote.
MIN_FRAME_STDDEV = 8.0
# The detect loop is paced (see DETECT_LOOP_MIN_CYCLE below) so one add() call
# happens per cycle whether OCR ran or not — the vote rate is uniform, so a
# small window like 8 covers ~1.6 s of decision history evenly.

# Detect loop pacing: minimum seconds per iteration. OCR takes ~200 ms and
# hits this naturally; the fast no-quad path is padded to match, so we don't
# flood the Confirmer with rapid `None` votes that would age real matches
# out of the window before their partner arrives.
DETECT_LOOP_MIN_CYCLE = 0.2

# OCR / matching
# Engine for reading the name strip:
#   "readtext"  — full EasyOCR: LOCATES the text within the crop first, so it
#                 reliably reads real (noisy, textured) card strips. This is the
#                 one that actually works on a live camera. We now run it only
#                 ONCE per frame (on the single most-prominent card), so its cost
#                 is paid once, not per detected box.
#   "recognize" — recognizer only (~7 ms): fast, but it reads the whole crop as
#                 one line and tends to return nothing on real/messy strips — it
#                 only behaves on a perfectly tight, clean crop. Use at your own
#                 risk for speed.
OCR_ENGINE = "readtext"
OCR_MIN_CONFIDENCE = 0.25        # Minimum EasyOCR confidence to use a result
FUZZY_MATCH_THRESHOLD = 72       # RapidFuzz score (0-100) to accept a name match
MAX_DETECTED_DISPLAY = 8         # Max simultaneously-shown detected cards in UI

# Live Scryfall fallback: when the local fuzzy match fails, the matcher
# hits Scryfall's /cards/named?fuzzy= endpoint synchronously (one HTTP
# call), caches the result in memory, and appends any new hit to
# CARD_NAMES_GROWN_FILE so we never re-fetch it. Rate-limited to be
# polite to Scryfall (they recommend ≥50 ms between calls; we use 120).
LIVE_MATCH_MIN_INTERVAL = 0.12   # seconds between successive live Scryfall calls
LIVE_MATCH_TIMEOUT = 5.0         # sync HTTP timeout for the live fallback

# Server
HOST = "127.0.0.1"
PORT = 8000
