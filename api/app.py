"""
FastAPI application.

Routes:
  GET  /                          → main UI (index.html)
  GET  /video                     → MJPEG video stream
  GET  /api/cards                 → full decklist JSON
  POST /api/cards                 → add card to deck
  PATCH /api/cards/{name}         → set count / move zone
  DELETE /api/cards/{name}        → remove card (or decrement)
  GET  /api/search?q=…            → search Scryfall local index
  GET  /api/card/{name}           → fetch single card data
  POST /api/deck/clear            → wipe deck (main/side/both)
  GET  /api/export/{fmt}          → export decklist text
  WS   /ws                        → real-time card-detection events
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from decklist.manager import Zone, decklist
from scanner.detector import CardScanner, list_cameras
from scryfall.client import scryfall

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent.parent / "static"

# ---------------------------------------------------------------------------
# Globals initialised on startup
# ---------------------------------------------------------------------------
scanner: Optional[CardScanner] = None
_ws_clients: list[WebSocket] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global scanner

    # Load Scryfall data in background so the UI is immediately available
    asyncio.create_task(_load_scryfall())

    # Start the video scanner thread
    scanner = CardScanner(card_names=[])
    scanner.start()

    # Background task: relay card detection events to WebSocket clients
    asyncio.create_task(_relay_card_events())

    yield

    if scanner:
        scanner.stop()


app = FastAPI(title="MTG Card Scanner", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


async def _load_scryfall() -> None:
    try:
        await scryfall.ensure_bulk_loaded()
        if scanner:
            scanner.update_card_names(scryfall.all_names())
        log.info("Scryfall data loaded and scanner updated.")
    except Exception as exc:
        log.error("Failed to load Scryfall data: %s", exc)


async def _relay_card_events() -> None:
    """Forward card-detection events from the scanner queue to all WS clients."""
    loop = asyncio.get_event_loop()
    while True:
        if not scanner:
            await asyncio.sleep(0.5)
            continue
        try:
            # Poll the queue without blocking the event loop
            cards = await loop.run_in_executor(
                None, lambda: scanner.card_event_queue.get(timeout=0.3)
            )
        except Exception:
            await asyncio.sleep(0.05)
            continue

        if not _ws_clients:
            continue

        payload = json.dumps({
            "type": "detected",
            "cards": [
                {
                    "raw": c.raw_ocr_text,
                    "name": c.matched_name,
                    "confidence": round(c.confidence, 3),
                }
                for c in cards
                if c.matched_name
            ],
        })

        dead: list[WebSocket] = []
        for ws in _ws_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


# ---------------------------------------------------------------------------
# HTML entry-point
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# MJPEG video stream
# ---------------------------------------------------------------------------

@app.get("/video")
async def video_feed() -> StreamingResponse:
    if not scanner:
        raise HTTPException(503, "Scanner not yet initialised")

    def frame_generator():
        yield from scanner.latest_jpeg()

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------------------------
# Decklist API
# ---------------------------------------------------------------------------

class AddCardRequest(BaseModel):
    name: str
    count: int = Field(default=1, ge=1, le=99)
    zone: Zone = "main"


class PatchCardRequest(BaseModel):
    count: Optional[int] = Field(default=None, ge=0, le=99)
    zone: Optional[Zone] = None


@app.get("/api/cards")
async def get_cards():
    entries = decklist.list_entries()
    return {
        **entries,
        "total_main": decklist.total_cards("main"),
        "total_side": decklist.total_cards("side"),
    }


@app.post("/api/cards")
async def add_card(req: AddCardRequest):
    # Fetch card data from local index
    card_data = scryfall.get_card(req.name)
    if not card_data:
        # Try a live lookup for very new cards
        card_data = await scryfall.fetch_card_live(req.name)

    if not card_data:
        # Allow adding even if not in Scryfall (user might know the name)
        entry = decklist.add(req.name, req.count, req.zone)
    else:
        entry = decklist.add(
            card_data.name,
            req.count,
            req.zone,
            set_code=card_data.set_code,
            collector_number=card_data.collector_number,
            image_uri=card_data.image_uri,
            type_line=card_data.type_line,
            mana_cost=card_data.mana_cost,
        )

    await _broadcast_deck_update()
    return entry.to_dict()


@app.patch("/api/cards/{name}")
async def patch_card(name: str, req: PatchCardRequest):
    # Auto-detect zone when caller didn't specify
    if req.zone is not None:
        zone: Zone = req.zone
    elif decklist.get_entry(name, "side") and not decklist.get_entry(name, "main"):
        zone = "side"
    else:
        zone = "main"

    if req.count is not None:
        entry = decklist.set_count(name, req.count, zone)
    else:
        entry = decklist.get_entry(name, zone)

    await _broadcast_deck_update()
    return entry.to_dict() if entry else {"removed": True}


@app.delete("/api/cards/{name}")
async def remove_card(
    name: str,
    count: int = Query(default=1, ge=1),
    zone: Zone = Query(default="main"),
):
    decklist.remove(name, count, zone)
    await _broadcast_deck_update()
    return {"ok": True}


@app.post("/api/deck/clear")
async def clear_deck(zone: Optional[Zone] = Query(default=None)):
    decklist.clear(zone)
    await _broadcast_deck_update()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@app.get("/api/export/{fmt}")
async def export_deck(fmt: str):
    valid = ("text", "moxfield", "mtga", "mtgo", "arena")
    if fmt not in valid:
        raise HTTPException(400, f"Unknown format. Valid: {valid}")
    text = decklist.export(fmt)  # type: ignore[arg-type]
    return Response(content=text, media_type="text/plain; charset=utf-8")


# ---------------------------------------------------------------------------
# Camera management
# ---------------------------------------------------------------------------

class SelectCameraRequest(BaseModel):
    source: int


@app.get("/api/cameras")
async def get_cameras():
    """List all available capture devices with human-readable names."""
    cameras = list_cameras()
    current = scanner.current_source if scanner else 0
    return {"cameras": cameras, "current": current}


@app.post("/api/cameras/select")
async def select_camera(req: SelectCameraRequest):
    if not scanner:
        raise HTTPException(503, "Scanner not initialised")
    scanner.switch_source(req.source)
    return {"ok": True, "source": req.source}


# ---------------------------------------------------------------------------
# Scryfall search
# ---------------------------------------------------------------------------

@app.get("/api/search")
async def search_cards(q: str = Query(default="", min_length=1)):
    results = scryfall.search(q, limit=20)
    return {"results": [c.to_dict() for c in results]}


@app.get("/api/card/{name:path}")
async def get_card(name: str):
    card = scryfall.get_card(name)
    if not card:
        card = await scryfall.fetch_card_live(name)
    if not card:
        raise HTTPException(404, f"Card not found: {name!r}")
    return card.to_dict()


# ---------------------------------------------------------------------------
# WebSocket
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    _ws_clients.append(ws)
    log.info("WS client connected (%d total)", len(_ws_clients))
    try:
        while True:
            # Keep the connection alive; client can also send messages
            data = await ws.receive_text()
            msg = json.loads(data)

            # Client can request adding a card via WS too
            if msg.get("action") == "add_card":
                name = msg.get("name", "")
                count = int(msg.get("count", 1))
                zone = msg.get("zone", "main")
                if name:
                    card_data = scryfall.get_card(name)
                    if card_data:
                        decklist.add(
                            card_data.name, count, zone,
                            set_code=card_data.set_code,
                            collector_number=card_data.collector_number,
                            image_uri=card_data.image_uri,
                            type_line=card_data.type_line,
                            mana_cost=card_data.mana_cost,
                        )
                    else:
                        decklist.add(name, count, zone)
                    await _broadcast_deck_update()

    except WebSocketDisconnect:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)
        log.info("WS client disconnected (%d remaining)", len(_ws_clients))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _broadcast_deck_update() -> None:
    if not _ws_clients:
        return
    entries = decklist.list_entries()
    payload = json.dumps({
        "type": "deck_update",
        "main": entries["main"],
        "side": entries["side"],
        "total_main": decklist.total_cards("main"),
        "total_side": decklist.total_cards("side"),
    })
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.remove(ws)
