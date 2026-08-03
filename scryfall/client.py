from __future__ import annotations

import asyncio
import gzip
import json
import logging
import re
import threading
import time
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

from config import (
    CARD_NAMES_FILE,
    CARD_NAMES_GROWN_FILE,
    FUZZY_MATCH_THRESHOLD,
    LIVE_MATCH_MIN_INTERVAL,
    LIVE_MATCH_TIMEOUT,
    SCRYFALL_BULK_DATA_URL,
    SCRYFALL_SEARCH_URL,
)

log = logging.getLogger(__name__)

BULK_DATA_TTL_DAYS = 3
_CACHE_META_FILE = CARD_NAMES_FILE.parent / "bulk_meta.json"


def _warn_if_rate_limited(resp, context: str) -> None:
    """Emit a WARNING when Scryfall returns 429 so it's obvious at the
    console. Includes the Retry-After header when present."""
    if getattr(resp, "status_code", None) == 429:
        retry = resp.headers.get("retry-after", "?")
        log.warning("SCRYFALL RATE LIMITED (429) on %s — retry-after=%s", context, retry)

# Match anything that's not a letter or digit — used to build a punctuation-
# and whitespace-free search key so users can type "atraxa praetors voice" (or
# "atraxapraetorsvoice") and still hit "Atraxa, Praetor's Voice".
_NON_ALNUM = re.compile(r"[^a-z0-9]")


def _normalize_for_search(s: str) -> str:
    """Lowercase + strip diacritics + drop everything that isn't a letter/digit."""
    nfd = unicodedata.normalize("NFD", s)
    no_marks = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return _NON_ALNUM.sub("", no_marks.lower())


@dataclass
class CardData:
    name: str
    mana_cost: str
    type_line: str
    oracle_text: str
    set_code: str
    collector_number: str
    rarity: str
    image_uri: str
    scryfall_uri: str
    colors: list[str]
    cmc: float
    legalities: dict[str, str]

    def to_dict(self) -> dict:
        return asdict(self)


def _parse_card(raw: dict) -> Optional[CardData]:
    if raw.get("layout") in {"token", "double_faced_token", "art_series", "emblem"}:
        return None
    if raw.get("object") != "card":
        return None

    name = raw.get("name", "")
    if not name:
        return None

    # DFC / split cards: pick front face image and mana cost
    image_uri = ""
    faces = raw.get("card_faces")
    if faces:
        image_uri = faces[0].get("image_uris", {}).get("normal", "")
    if not image_uri:
        image_uri = raw.get("image_uris", {}).get("normal", "")

    return CardData(
        name=name,
        mana_cost=raw.get("mana_cost") or (faces[0].get("mana_cost") if faces else "") or "",
        type_line=raw.get("type_line", ""),
        oracle_text=raw.get("oracle_text") or (faces[0].get("oracle_text") if faces else "") or "",
        set_code=raw.get("set", ""),
        collector_number=raw.get("collector_number", ""),
        rarity=raw.get("rarity", ""),
        image_uri=image_uri,
        scryfall_uri=raw.get("scryfall_uri", ""),
        colors=raw.get("colors", []),
        cmc=raw.get("cmc", 0.0),
        legalities=raw.get("legalities", {}),
    )


class ScryfallClient:
    def __init__(self) -> None:
        self._index: dict[str, CardData] = {}
        self._match_keys: list[str] = []   # lowercase names, cached for fuzzy matching
        # Punctuation-stripped lookup table for the user-facing substring search.
        # Built once after load so each query is one normalize + a linear scan.
        self._search_keys: list[tuple[str, CardData]] = []
        self._loaded = False
        self._load_lock = asyncio.Lock()

        # ── Live-fallback state (all guarded by _live_lock) ─────────────
        # _live_cache: raw_ocr_lowercased_stripped → canonical name | None
        #   None caches "we asked Scryfall and it also didn't know" so garbage
        #   OCR text can't spam repeated live calls.
        # _last_live_call: monotonic timestamp of the most recent HTTP call
        #   to Scryfall; used to enforce LIVE_MATCH_MIN_INTERVAL.
        # _grown: names (lowercased) already added to CARD_NAMES_GROWN_FILE;
        #   avoids duplicate append passes when the same live hit re-fires.
        self._live_cache: dict[str, Optional[str]] = {}
        self._last_live_call = 0.0
        self._grown: set[str] = set()
        self._live_lock = threading.Lock()

    async def ensure_bulk_loaded(self) -> None:
        async with self._load_lock:
            if self._loaded:
                return
            if self._is_cache_fresh():
                log.info("Loading card names from local cache…")
                self._load_from_cache()
            else:
                log.info("Downloading Scryfall bulk data (this is a one-time ~30 MB download)…")
                await self._download_and_cache()
            grown_count = self._load_grown_cache()
            self._match_keys = list(self._index.keys())
            self._search_keys = [
                (_normalize_for_search(card.name), card)
                for card in self._index.values()
            ]
            self._loaded = True
            log.info("Scryfall index ready: %d cards (%d from live-grown cache).",
                     len(self._index), grown_count)

    def _is_cache_fresh(self) -> bool:
        if not CARD_NAMES_FILE.exists():
            return False
        if not _CACHE_META_FILE.exists():
            return False
        try:
            meta = json.loads(_CACHE_META_FILE.read_text())
            age_days = (time.time() - meta["downloaded_at"]) / 86400
            return age_days < BULK_DATA_TTL_DAYS
        except Exception:
            return False

    def _load_from_cache(self) -> None:
        raw_list: list[dict] = json.loads(CARD_NAMES_FILE.read_text(encoding="utf-8"))
        for d in raw_list:
            card = CardData(**d)
            self._index[card.name.lower()] = card

    def _load_grown_cache(self) -> int:
        """Load any cards that previous sessions added via the live fallback.
        Returns the count added to the index (excluding names already in bulk)."""
        if not CARD_NAMES_GROWN_FILE.exists():
            return 0
        try:
            raw_list: list[dict] = json.loads(
                CARD_NAMES_GROWN_FILE.read_text(encoding="utf-8"))
        except Exception:
            log.warning("Grown cache at %s is unreadable; ignoring.", CARD_NAMES_GROWN_FILE)
            return 0
        added = 0
        for d in raw_list:
            try:
                card = CardData(**d)
            except Exception:
                continue
            key = card.name.lower()
            self._grown.add(key)
            if key not in self._index:
                self._index[key] = card
                added += 1
        return added

    def _append_grown_card_to_disk(self, card: CardData) -> None:
        """Persist a newly-discovered card so the next session already knows it.
        Rewrites the whole file each time — grown lists stay small and this
        avoids partial-write corruption from JSONL-style appends."""
        try:
            existing: list[dict] = []
            if CARD_NAMES_GROWN_FILE.exists():
                existing = json.loads(CARD_NAMES_GROWN_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = []
        existing.append(card.to_dict())
        tmp = CARD_NAMES_GROWN_FILE.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")
        tmp.replace(CARD_NAMES_GROWN_FILE)

    async def _download_and_cache(self) -> None:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(SCRYFALL_BULK_DATA_URL)
            _warn_if_rate_limited(resp, "bulk-data manifest")
            resp.raise_for_status()
            manifest = resp.json()
            # Scryfall deprecated `download_uri` (single JSON array) in favour
            # of `jsonl_download_uri` (streaming JSONL, one object per line).
            # Fall back for older Scryfall responses if `download_uri` shows
            # up again in the future.
            download_url = manifest.get("jsonl_download_uri") or manifest.get("download_uri")
            if not download_url:
                raise RuntimeError(
                    "Scryfall bulk-data manifest has no download URI "
                    f"(keys: {sorted(manifest.keys())})"
                )
            is_jsonl = ".jsonl" in download_url
            is_gzip  = download_url.endswith(".gz")

            log.info("Fetching %s …", download_url)
            async with client.stream("GET", download_url) as stream:
                _warn_if_rate_limited(stream, "bulk-data payload")
                stream.raise_for_status()
                data = await stream.aread()

        # Scryfall serves the JSONL bulk gzipped with `Content-Type:
        # application/gzip` (NOT `Content-Encoding: gzip`), so httpx does
        # not auto-decompress. Do it here.
        if is_gzip:
            data = gzip.decompress(data)

        cards_raw: list[dict]
        if is_jsonl:
            cards_raw = [json.loads(line) for line in data.splitlines() if line.strip()]
        else:
            cards_raw = json.loads(data)

        cards: list[CardData] = []
        for raw in cards_raw:
            card = _parse_card(raw)
            if card:
                self._index[card.name.lower()] = card
                cards.append(card)

        CARD_NAMES_FILE.write_text(
            json.dumps([c.to_dict() for c in cards], ensure_ascii=False),
            encoding="utf-8",
        )
        _CACHE_META_FILE.write_text(
            json.dumps({"downloaded_at": time.time(), "count": len(cards)}),
            encoding="utf-8",
        )
        log.info("Cached %d cards to %s", len(cards), CARD_NAMES_FILE)

    def get_card(self, name: str) -> Optional[CardData]:
        return self._index.get(name.lower())

    def search(self, query: str, limit: int = 20) -> list[CardData]:
        """Punctuation/diacritic-insensitive substring search over card names.

        Both the query and the index keys are normalised to lowercase letters
        and digits only, so "praetors voice", "praetor's voice", and
        "praetorsvoice" all match "Atraxa, Praetor's Voice". Cards whose
        normalised name STARTS with the query are listed first.
        """
        q = _normalize_for_search(query)
        if not q:
            return []
        hits = [(key, card) for key, card in self._search_keys if q in key]
        hits.sort(key=lambda kc: (not kc[0].startswith(q), kc[1].name))
        return [card for _, card in hits[:limit]]

    def all_names(self) -> list[str]:
        return [c.name for c in self._index.values()]

    def match_ocr_name(
        self, raw_text: str, threshold: Optional[int] = None
    ) -> tuple[Optional[str], float]:
        """Fuzzy-match raw OCR text to a card and return (canonical_name, score).

        Two-stage: first local RapidFuzz over the bulk cache (fast, offline);
        if that returns nothing, sync-fallback to Scryfall's /cards/named?fuzzy
        endpoint. Successful live hits are added to the in-memory index AND
        appended to CARD_NAMES_GROWN_FILE so subsequent sessions know them.

        The scanner votes on the resolved name, so a NEW live hit is treated
        the same as a local hit — the Confirmer's M-of-N vote naturally keeps
        us from emitting on a single Scryfall guess.
        """
        if not raw_text or not raw_text.strip():
            return None, 0.0

        local_name, local_score = self._local_match(raw_text, threshold)
        if local_name is not None:
            return local_name, local_score

        # Local miss — check the negative cache before spending an HTTP call.
        key = raw_text.strip().lower()
        with self._live_lock:
            if key in self._live_cache:
                cached = self._live_cache[key]
                return (cached, 0.9 if cached else 0.0)

        return self._live_match_sync(raw_text, key)

    def _local_match(
        self, raw_text: str, threshold: Optional[int]
    ) -> tuple[Optional[str], float]:
        """The offline half of match_ocr_name — RapidFuzz over the bulk cache."""
        from rapidfuzz import fuzz, process  # noqa: PLC0415

        keys = self._match_keys or list(self._index.keys())
        if not keys:
            return None, 0.0
        th = FUZZY_MATCH_THRESHOLD if threshold is None else threshold
        match = process.extractOne(
            raw_text.lower(), keys, scorer=fuzz.WRatio, score_cutoff=th,
        )
        if not match:
            return None, 0.0
        card = self._index.get(match[0])
        return (card.name if card else None), match[1] / 100.0

    def _live_match_sync(
        self, raw_text: str, cache_key: str
    ) -> tuple[Optional[str], float]:
        """Blocking Scryfall fuzzy lookup. Rate-limited and negative-cached.

        Called from the scanner's detect thread (not from the async event
        loop) so a sync ``httpx.Client`` is the natural choice. One call
        costs ~200-500 ms; the detect loop's DETECT_LOOP_MIN_CYCLE=0.2
        already budgets for that."""
        with self._live_lock:
            wait = LIVE_MATCH_MIN_INTERVAL - (time.monotonic() - self._last_live_call)
            if wait > 0:
                time.sleep(wait)
            self._last_live_call = time.monotonic()

        try:
            with httpx.Client(timeout=LIVE_MATCH_TIMEOUT) as client:
                resp = client.get(SCRYFALL_SEARCH_URL, params={"fuzzy": raw_text})
        except Exception as exc:
            log.warning("Live match HTTP error for %r: %s", raw_text, exc)
            # Do NOT negative-cache transient network errors — a retry next
            # cycle might succeed.
            return None, 0.0

        _warn_if_rate_limited(resp, f"live match {raw_text!r}")

        if resp.status_code == 200:
            try:
                card = _parse_card(resp.json())
            except Exception as exc:
                log.warning("Live match parse error for %r: %s", raw_text, exc)
                return None, 0.0
            if card is None:
                # Scryfall returned a non-card object (token, art_series, …).
                with self._live_lock:
                    self._live_cache[cache_key] = None
                return None, 0.0

            self._grow_index_with(card)
            with self._live_lock:
                self._live_cache[cache_key] = card.name
            log.info("live match: %r → %r (grew cache; %d cards)",
                     raw_text, card.name, len(self._index))
            return card.name, 0.9   # score-equivalent — arbitrary but above threshold

        # 404 (Scryfall doesn't know) OR any other non-200 → negative cache
        # so we don't re-query on every subsequent OCR read of the same text.
        with self._live_lock:
            self._live_cache[cache_key] = None
        log.info("live miss: %r (Scryfall HTTP %d; negative-cached)",
                 raw_text, resp.status_code)
        return None, 0.0

    def _grow_index_with(self, card: CardData) -> None:
        """Add a new card to the in-memory index AND persist to disk."""
        key = card.name.lower()
        if key not in self._index:
            self._index[key] = card
            self._match_keys.append(key)
            self._search_keys.append((_normalize_for_search(card.name), card))
        with self._live_lock:
            already_persisted = key in self._grown
            if not already_persisted:
                self._grown.add(key)
        if not already_persisted:
            try:
                self._append_grown_card_to_disk(card)
            except Exception as exc:
                log.warning("Could not persist grown card %r: %s", card.name, exc)

    def match_ocr(self, raw_text: str, threshold: Optional[int] = None) -> Optional[CardData]:
        """Fuzzy-match raw OCR text to a real card and return its data.

        This is the deferred lookup the detection pipeline hands off to: the
        scanner only confirms a steady OCR *string*; resolving it to an actual
        card (the ~7-24 ms fuzzy search over ~34k names) happens here, off the
        detection thread. Call it from a thread/executor — it is CPU-bound.
        """
        from rapidfuzz import fuzz, process  # noqa: PLC0415

        if not raw_text or not raw_text.strip():
            return None
        keys = self._match_keys or list(self._index.keys())
        if not keys:
            return None
        th = FUZZY_MATCH_THRESHOLD if threshold is None else threshold
        match = process.extractOne(
            raw_text.lower(), keys, scorer=fuzz.WRatio, score_cutoff=th,
        )
        if not match:
            return None
        return self._index.get(match[0])

    async def fetch_card_live(self, name: str) -> Optional[CardData]:
        """Live Scryfall lookup for cards not yet in the local cache."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    SCRYFALL_SEARCH_URL,
                    params={"fuzzy": name},
                )
                _warn_if_rate_limited(resp, f"fetch_card_live({name!r})")
                if resp.status_code == 200:
                    card = _parse_card(resp.json())
                    if card:
                        self._index[card.name.lower()] = card
                    return card
                if resp.status_code not in (200, 404):
                    log.warning("Scryfall fetch_card_live(%r) → HTTP %d", name, resp.status_code)
        except Exception as exc:
            log.warning("Live Scryfall lookup failed for %r: %s", name, exc)
        return None


scryfall = ScryfallClient()
