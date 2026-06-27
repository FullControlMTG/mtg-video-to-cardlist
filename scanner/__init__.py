"""Card-scanner package.

Public surface — what callers should import:
  - `CardScanner`  : the camera+detect+stream orchestrator
  - `Matcher`      : type alias for the injected raw-OCR → name resolver
  - `Detection`    : a confirmed scan result
  - `list_cameras` : enumerate plugged-in webcams
  - `prewarm_ocr`  : kick off EasyOCR initialisation off the request path
"""

from .cameras import list_cameras
from .confirmer import Detection
from .detector import CardScanner, Matcher
from .ocr import prewarm_ocr

__all__ = ["CardScanner", "Detection", "Matcher", "list_cameras", "prewarm_ocr"]
