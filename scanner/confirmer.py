"""Per-frame card-name vote that confirms when the same name wins M of
the last N reads.

Decoupled from the rest of the scanner so it can be tested with synthetic
sequences and so the confirmation policy can evolve independently from
OCR/detection.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional

from config import CONFIRM_MIN_MATCH, CONFIRM_WINDOW_SIZE


@dataclass
class Detection:
    """A confirmed card — the resolved Scryfall name and its vote confidence."""
    name: str
    confidence: float


class Confirmer:
    """Confirms the card by voting on the FUZZY-MATCHED card NAME over recent
    frames — globally, not per detected box.

    Why global: a hand-held stack jitters, so keying votes to a box position
    fragments them across many short-lived tracks and nothing ever reaches the
    threshold. Since the user presents one card at a time, a single rolling vote
    over the last CONFIRM_WINDOW_SIZE (N) frames is robust to that jitter: each
    frame contributes the best matched name (or None), and a name is confirmed
    once it wins CONFIRM_MIN_MATCH (M) of those frames. Voting on the matched
    name (not raw OCR) means noisy reads ("Lightning Bo1t") still count toward
    the right card. A different card taking over re-confirms; an empty window
    (card removed) clears the state so it can re-confirm later.
    """

    def __init__(self) -> None:
        self.recent: deque[Optional[str]] = deque(maxlen=CONFIRM_WINDOW_SIZE)
        self.candidate: Optional[str] = None   # current best guess (for overlay)
        self.confirmed: Optional[str] = None   # currently confirmed name (green)
        self.confidence = 0.0
        self._emitted: Optional[str] = None

    def add(self, name: Optional[str]) -> Optional[str]:
        """Record this frame's best matched name; return a name iff NEWLY confirmed."""
        self.recent.append(name)
        votes = Counter(n for n in self.recent if n)
        if not votes:
            self.candidate = self.confirmed = self._emitted = None
            self.confidence = 0.0
            return None

        top, count = votes.most_common(1)[0]
        self.candidate = top
        self.confidence = count / len(self.recent)

        # Forget a prior confirmation once that card stops appearing.
        if self._emitted and self._emitted not in votes:
            self._emitted = None
            self.confirmed = None

        if count >= CONFIRM_MIN_MATCH:
            self.confirmed = top
            if top != self._emitted:
                self._emitted = top
                return top
        return None
