import asyncio
import errno
import logging
import socket
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


def _port_in_use(host: str, port: int) -> bool:
    """Cheap pre-flight so we can print a clear message rather than dumping
    uvicorn's stack trace when the port is already bound. Sets SO_REUSEADDR
    to mirror uvicorn's own bind behaviour — without it, a socket lingering
    in TIME_WAIT from a just-stopped instance would falsely flag as in-use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError as exc:
            return exc.errno in (errno.EADDRINUSE, errno.EACCES)
        return False


def main() -> None:
    if _port_in_use(HOST, PORT):
        log.error(
            "Port %d on %s is already in use. Another instance of the "
            "scanner is probably running — stop it first (Ctrl+C in its "
            "terminal, or `lsof -tiTCP:%d | xargs kill`).",
            PORT, HOST, PORT,
        )
        sys.exit(1)

    log.info("Starting MTG Card Scanner on http://%s:%d", HOST, PORT)

    async def open_browser() -> None:
        try:
            await asyncio.sleep(1.5)
            webbrowser.open(f"http://{HOST}:{PORT}")
        except asyncio.CancelledError:
            return

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
    browser_task = loop.create_task(open_browser())
    try:
        loop.run_until_complete(server.serve())
    except KeyboardInterrupt:
        log.info("Shutting down.")
    finally:
        # Cancel any still-pending tasks (browser opener if the server was
        # short-lived; uvicorn's own tasks if shutdown was abrupt) so the
        # loop can close without "Task was destroyed while pending" warnings.
        browser_task.cancel()
        pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
        if pending:
            for t in pending:
                t.cancel()
            try:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            except Exception:
                pass
        loop.close()


if __name__ == "__main__":
    main()
