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

FORMATS = [
    "Standard", "Pioneer", "Modern", "Legacy", "Vintage",
    "Commander", "Pauper", "Penny Dreadful", "Historic", "Alchemy",
    "Brawl", "Historic Brawl", "Oathbreaker", "Premodern", "Old School",
]


class DeckMetadata:
    __slots__ = ("name", "format", "commander", "notes")

    def __init__(
        self,
        name: str = "My Deck",
        format: str = "",       # noqa: A002
        commander: str = "",
        notes: str = "",
    ) -> None:
        self.name      = name
        self.format    = format
        self.commander = commander
        self.notes     = notes

    def to_dict(self) -> dict:
        return {
            "name":      self.name,
            "format":    self.format,
            "commander": self.commander,
            "notes":     self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DeckMetadata":
        return cls(
            name      = d.get("name", "My Deck"),
            format    = d.get("format", ""),
            commander = d.get("commander", ""),
            notes     = d.get("notes", ""),
        )

    @property
    def is_commander_format(self) -> bool:
        return self.format.lower() in ("commander", "brawl", "historic brawl", "oathbreaker")


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
        self._main: OrderedDict[str, DeckEntry] = OrderedDict()
        self._side: OrderedDict[str, DeckEntry] = OrderedDict()
        self._meta: DeckMetadata = DeckMetadata()
        self._load()

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
            if "meta" in data:
                self._meta = DeckMetadata.from_dict(data["meta"])
            log.info("Loaded decklist: %d main, %d side.", len(self._main), len(self._side))
        except Exception as exc:
            log.warning("Could not load decklist: %s", exc)

    def _save(self) -> None:
        try:
            data = {
                "meta": self._meta.to_dict(),
                "main": [e.to_dict() for e in self._main.values()],
                "side": [e.to_dict() for e in self._side.values()],
            }
            DECKLIST_FILE.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        except Exception as exc:
            log.warning("Could not save decklist: %s", exc)

    def get_meta(self) -> DeckMetadata:
        return self._meta

    def set_meta(self, **kwargs) -> DeckMetadata:
        for k, v in kwargs.items():
            if hasattr(self._meta, k):
                setattr(self._meta, k, v)
        self._save()
        return self._meta

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

    def export(self, fmt: ExportFormat = "text") -> str:
        if fmt in ("text", "moxfield", "mtgo"):
            return self._export_plain(fmt)
        if fmt in ("mtga", "arena"):
            return self._export_arena()
        raise ValueError(f"Unknown export format: {fmt!r}")

    def _meta_comments(self) -> list[str]:
        lines = []
        if self._meta.name:
            lines.append(f"// Name: {self._meta.name}")
        if self._meta.format:
            lines.append(f"// Format: {self._meta.format}")
        if self._meta.commander and self._meta.is_commander_format:
            lines.append(f"// Commander: {self._meta.commander}")
        if self._meta.notes:
            for note_line in self._meta.notes.splitlines():
                lines.append(f"// {note_line}")
        return lines

    def _export_plain(self, fmt: ExportFormat) -> str:
        lines: list[str] = self._meta_comments()
        if lines:
            lines.append("")

        if self._meta.is_commander_format and self._meta.commander:
            lines.append("Commander:")
            lines.append(f"1 {self._meta.commander}")
            lines.append("")

        for entry in self._main.values():
            if self._meta.is_commander_format and entry.name == self._meta.commander:
                continue
            lines.append(f"{entry.count} {entry.name}")

        if self._side:
            lines.append("")
            lines.append("Sideboard:")
            for entry in self._side.values():
                lines.append(f"{entry.count} {entry.name}")

        return "\n".join(lines)

    def _export_arena(self) -> str:
        def card_line(entry: DeckEntry) -> str:
            if entry.set_code and entry.collector_number:
                return f"{entry.count} {entry.name} ({entry.set_code.upper()}) {entry.collector_number}"
            return f"{entry.count} {entry.name}"

        lines: list[str] = self._meta_comments()
        if lines:
            lines.append("")

        if self._meta.is_commander_format and self._meta.commander:
            lines.append("Commander")
            lines.append(f"1 {self._meta.commander}")
            lines.append("")

        lines.append("Deck")
        for entry in self._main.values():
            if self._meta.is_commander_format and entry.name == self._meta.commander:
                continue
            lines.append(card_line(entry))

        if self._side:
            lines.append("")
            lines.append("Sideboard")
            for entry in self._side.values():
                lines.append(card_line(entry))

        return "\n".join(lines)


decklist = DecklistManager()
