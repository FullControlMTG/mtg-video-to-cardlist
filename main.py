"""
MTG Card Scanner — entry point.

Run with:
    python main.py

Or directly:
    uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload
"""

import asyncio
import logging
import sys
import webbrowser

import uvicorn

from config import HOST, PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mtg_scanner")


def main() -> None:
    log.info("Starting MTG Card Scanner on http://%s:%d", HOST, PORT)

    # Open browser after a short delay so the server has time to bind
    async def open_browser() -> None:
        await asyncio.sleep(1.5)
        webbrowser.open(f"http://{HOST}:{PORT}")

    config = uvicorn.Config(
        "api.app:app",
        host=HOST,
        port=PORT,
        log_level="info",
        loop="asyncio",
    )
    server = uvicorn.Server(config)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.create_task(open_browser())
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        loop.close()


if __name__ == "__main__":
    main()
