from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx

from config import (
    CARD_NAMES_FILE,
    SCRYFALL_BULK_DATA_URL,
    SCRYFALL_SEARCH_URL,
)

log = logging.getLogger(__name__)

BULK_DATA_TTL_DAYS = 3
_CACHE_META_FILE = CARD_NAMES_FILE.parent / "bulk_meta.json"


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
        self._loaded = False
        self._load_lock = asyncio.Lock()

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
            self._loaded = True
            log.info("Scryfall index ready: %d cards.", len(self._index))

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

    async def _download_and_cache(self) -> None:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.get(SCRYFALL_BULK_DATA_URL)
            resp.raise_for_status()
            download_url = resp.json()["download_uri"]

            log.info("Fetching %s …", download_url)
            async with client.stream("GET", download_url) as stream:
                stream.raise_for_status()
                data = await stream.aread()

        cards_raw: list[dict] = json.loads(data)
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
        q = query.lower().strip()
        if not q:
            return []
        results = [card for key, card in self._index.items() if q in key]
        results.sort(key=lambda c: (not c.name.lower().startswith(q), c.name))
        return results[:limit]

    def all_names(self) -> list[str]:
        return [c.name for c in self._index.values()]

    async def fetch_card_live(self, name: str) -> Optional[CardData]:
        """Live Scryfall lookup for cards not yet in the local cache."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    SCRYFALL_SEARCH_URL,
                    params={"fuzzy": name},
                )
                if resp.status_code == 200:
                    card = _parse_card(resp.json())
                    if card:
                        self._index[card.name.lower()] = card
                    return card
        except Exception as exc:
            log.warning("Live Scryfall lookup failed for %r: %s", name, exc)
        return None


scryfall = ScryfallClient()
