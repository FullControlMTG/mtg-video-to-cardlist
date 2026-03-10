"""
Decklist management and export.

The decklist is split into a main deck and a sideboard.
All state is persisted to DECKLIST_FILE automatically.

Export formats:
  - text      : "4 Lightning Bolt" (universal)
  - moxfield  : same as text (Moxfield accepts plain text)
  - mtga      : Arena format with set/collector number where available
  - mtgo      : MTGO .dek compatible plain text
  - arena     : explicit "Deck" / "Sideboard" headers for Arena import
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Optional

from config import DECKLIST_FILE

log = logging.getLogger(__name__)

Zone = Literal["main", "side"]
ExportFormat = Literal["text", "moxfield", "mtga", "mtgo", "arena"]


class DeckEntry:
    __slots__ = ("name", "count", "set_code", "collector_number", "image_uri", "type_line", "mana_cost")

    def __init__(
        self,
        name: str,
        count: int = 1,
        set_code: str = "",
        collector_number: str = "",
        image_uri: str = "",
        type_line: str = "",
        mana_cost: str = "",
    ) -> None:
        self.name = name
        self.count = count
        self.set_code = set_code
        self.collector_number = collector_number
        self.image_uri = image_uri
        self.type_line = type_line
        self.mana_cost = mana_cost

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "count": self.count,
            "set_code": self.set_code,
            "collector_number": self.collector_number,
            "image_uri": self.image_uri,
            "type_line": self.type_line,
            "mana_cost": self.mana_cost,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeckEntry":
        return cls(**d)


class DecklistManager:
    def __init__(self) -> None:
        # Ordered dicts preserve insertion order for display
        self._main: OrderedDict[str, DeckEntry] = OrderedDict()
        self._side: OrderedDict[str, DeckEntry] = OrderedDict()
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not DECKLIST_FILE.exists():
            return
        try:
            data = json.loads(DECKLIST_FILE.read_text(encoding="utf-8"))
            for d in data.get("main", []):
                e = DeckEntry.from_dict(d)
                self._main[e.name] = e
            for d in data.get("side", []):
                e = DeckEntry.from_dict(d)
                self._side[e.name] = e
            log.info("Loaded decklist: %d main, %d side.", len(self._main), len(self._side))
        except Exception as exc:
            log.warning("Could not load decklist: %s", exc)

    def _save(self) -> None:
        try:
            data = {
                "main": [e.to_dict() for e in self._main.values()],
                "side": [e.to_dict() for e in self._side.values()],
            }
            DECKLIST_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            log.warning("Could not save decklist: %s", exc)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _zone(self, zone: Zone) -> OrderedDict[str, DeckEntry]:
        return self._main if zone == "main" else self._side

    def add(
        self,
        name: str,
        count: int = 1,
        zone: Zone = "main",
        *,
        set_code: str = "",
        collector_number: str = "",
        image_uri: str = "",
        type_line: str = "",
        mana_cost: str = "",
    ) -> DeckEntry:
        z = self._zone(zone)
        if name in z:
            z[name].count += count
        else:
            z[name] = DeckEntry(
                name=name,
                count=count,
                set_code=set_code,
                collector_number=collector_number,
                image_uri=image_uri,
                type_line=type_line,
                mana_cost=mana_cost,
            )
        self._save()
        return z[name]

    def remove(self, name: str, count: int = 1, zone: Zone = "main") -> Optional[DeckEntry]:
        z = self._zone(zone)
        entry = z.get(name)
        if not entry:
            return None
        entry.count -= count
        if entry.count <= 0:
            del z[name]
            self._save()
            return None
        self._save()
        return entry

    def set_count(self, name: str, count: int, zone: Zone = "main") -> Optional[DeckEntry]:
        z = self._zone(zone)
        if count <= 0:
            z.pop(name, None)
            self._save()
            return None
        if name in z:
            z[name].count = count
        else:
            z[name] = DeckEntry(name=name, count=count)
        self._save()
        return z[name]

    def clear(self, zone: Optional[Zone] = None) -> None:
        if zone is None or zone == "main":
            self._main.clear()
        if zone is None or zone == "side":
            self._side.clear()
        self._save()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_entry(self, name: str, zone: Zone = "main") -> Optional[DeckEntry]:
        return self._zone(zone).get(name)

    def list_entries(self, zone: Optional[Zone] = None) -> dict[str, list[dict]]:
        if zone == "main":
            return {"main": [e.to_dict() for e in self._main.values()]}
        if zone == "side":
            return {"side": [e.to_dict() for e in self._side.values()]}
        return {
            "main": [e.to_dict() for e in self._main.values()],
            "side": [e.to_dict() for e in self._side.values()],
        }

    def total_cards(self, zone: Optional[Zone] = None) -> int:
        if zone == "main":
            return sum(e.count for e in self._main.values())
        if zone == "side":
            return sum(e.count for e in self._side.values())
        return sum(e.count for e in self._main.values()) + sum(e.count for e in self._side.values())

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self, fmt: ExportFormat = "text") -> str:
        if fmt in ("text", "moxfield", "mtgo"):
            return self._export_plain(fmt)
        if fmt in ("mtga", "arena"):
            return self._export_arena()
        raise ValueError(f"Unknown export format: {fmt!r}")

    def _export_plain(self, fmt: ExportFormat) -> str:
        """
        Universal plain-text format accepted by Moxfield, MTGO, Archidekt, etc.

        4 Lightning Bolt
        2 Island
        ...

        Sideboard:
        1 Tormod's Crypt
        """
        lines: list[str] = []

        for entry in self._main.values():
            lines.append(f"{entry.count} {entry.name}")

        if self._side:
            lines.append("")
            lines.append("Sideboard:")
            for entry in self._side.values():
                lines.append(f"{entry.count} {entry.name}")

        return "\n".join(lines)

    def _export_arena(self) -> str:
        """
        MTGA / Arena format:

        Deck
        4 Lightning Bolt (M21) 152
        2 Island (ANB) 114

        Sideboard
        1 Tormod's Crypt (M21) 269
        """
        lines: list[str] = ["Deck"]
        for entry in self._main.values():
            if entry.set_code and entry.collector_number:
                lines.append(
                    f"{entry.count} {entry.name} ({entry.set_code.upper()}) {entry.collector_number}"
                )
            else:
                lines.append(f"{entry.count} {entry.name}")

        if self._side:
            lines.append("")
            lines.append("Sideboard")
            for entry in self._side.values():
                if entry.set_code and entry.collector_number:
                    lines.append(
                        f"{entry.count} {entry.name} ({entry.set_code.upper()}) {entry.collector_number}"
                    )
                else:
                    lines.append(f"{entry.count} {entry.name}")

        return "\n".join(lines)


# Module-level singleton
decklist = DecklistManager()
