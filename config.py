from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CARD_NAMES_FILE = DATA_DIR / "card_names.json"
DECKLIST_FILE = DATA_DIR / "decklist.json"

# Scryfall
SCRYFALL_BULK_DATA_URL = "https://api.scryfall.com/bulk-data/oracle-cards"
SCRYFALL_SEARCH_URL = "https://api.scryfall.com/cards/named"
SCRYFALL_AUTOCOMPLETE_URL = "https://api.scryfall.com/cards/autocomplete"

# Video capture
VIDEO_SOURCE = 0          # Webcam index (0 = default). Can be a RTSP URL string.
FRAME_SKIP = 5            # Process every Nth frame for card detection
JPEG_QUALITY = 75         # MJPEG stream quality

# Card detection (contour-based)
MIN_CARD_AREA = 8000      # Pixels² — filters out small noise contours
CARD_ASPECT_MIN = 0.55    # Width/height ratio lower bound (standard card ~0.715)
CARD_ASPECT_MAX = 0.80    # Width/height ratio upper bound
NAME_ROW_FRACTION = 0.17  # Top fraction of card image used for name OCR
NAME_COL_FRACTION = 0.72  # Left fraction of name strip (excludes mana-cost symbols)

# OCR / matching
OCR_MIN_CONFIDENCE = 0.25        # Minimum EasyOCR confidence to use a result
FUZZY_MATCH_THRESHOLD = 72       # RapidFuzz score (0-100) to accept a name match
MAX_DETECTED_DISPLAY = 8         # Max simultaneously-shown detected cards in UI

# Server
HOST = "127.0.0.1"
PORT = 8000
