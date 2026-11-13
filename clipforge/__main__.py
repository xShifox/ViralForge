"""``python -m clipforge`` - start the ViralForge server."""

from __future__ import annotations

import argparse
import logging
import socket
import sys
import time
import webbrowser
from pathlib import Path

from . import bootstrap

bootstrap.bootstrap()

from .config import ROOT  # noqa: E402

# Module level, not just inside main(): the helpers that open the interface are
# module-level functions and reached for this.
log = logging.getLogger("clipforge")


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((host, port)) == 0


def _running_instance(host: str, port: int, timeout: float = 1.5) -> str | None:
    """The base URL of a healthy ViralForge already serving this port, else None.

    Starting a second server when one is already running is not harmless: both
    write to the same job directory, so they overwrite each other's job files,
    and the browser ends up pointed at whichever one it happened to open. So the
    port is asked what it is before assuming it is free.
    """
    import json
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(f"{url}/api/system/health", timeout=timeout) as response:
            if response.status != 200:
                return None
            payload = json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None
    # Confirm it is ViralForge and not some other program that happens to hold the
    # port: a different service would answer with something else entirely.
    if not isinstance(payload, dict) or not payload.get("ok"):
        return None
    if "ffmpeg_version" not in payload or "data_dir" not in payload:
        return None
    return url


def _find_port(host: str, preferred: int, tries: int = 12) -> int:
    for offset in range(tries):
        candidate = preferred + offset
        if not _port_in_use(host, candidate):
            return candidate
    return preferred


def _show_ui(args, url: str) -> None:
    """Open the interface the way this launch asked for.

    One place, so a launch can never open a window and a tab at the same time -
    which is what happened when the detached launcher and its child both opened
    a browser.

    The URL is suffixed with a cache-busting timestamp to ensure the WebView2
    loads the freshly compiled dist/ files and doesn't serve stale cached assets.
    """
    if args.window:
        from .window import describe, open_ui

        # Add cache-busting timestamp to bust WebView2 cache
        # The timestamp is stripped by the backend when serving static files
        import time
        cache_bust = f"?t={int(time.time())}"

        # Maximised is the default, not full screen: full screen also hides the
        # title bar and the taskbar, taking away the close, minimise and
        # maximise buttons and leaving only Alt+F4 to get out.
        log.info(describe(open_ui(url + cache_bust, fullscreen=args.fullscreen)))
    elif args.open:
        webbrowser.open(url)


def _run_detached(args, port: int, url: str) -> int:
    """Start the server in its own process group and hand back control.

    A pipeline can spend minutes in Whisper and again encoding every clip, and it
    dies with the process. Run in the foreground, closing the window that started
    it kills a job half-way through - which looks like a hang, because the browser
    has no way to tell a dead server from a slow one.
    """
    import subprocess

    log_dir = ROOT / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "clipforge.log"

    command = [
        sys.executable, "-m", "clipforge",
        "--host", args.host, "--port", str(port),
        "--log-level", args.log_level,
    ]
    # Deliberately no --open for the child. The parent opens the interface once
    # the port is confirmed up; passing --open as well made every launch open two
    # tabs, one from each end.

    # CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP: the server gets no console at
    # all, and still outlives the one that started it. DETACHED_PROCESS also
    # works, but leaves a console host process behind, and that is the black box
    # that flashes on the taskbar.
    if sys.platform == "win32":
        creationflags = 0x08000000 | 0x00000200   # CREATE_NO_WINDOW | NEW_PROCESS_GROUP
    else:
        creationflags = 0

    with log_file.open("ab") as handle:
        child = subprocess.Popen(
            command, cwd=str(ROOT), stdout=handle, stderr=handle,
            stdin=subprocess.DEVNULL, creationflags=creationflags,
            start_new_session=sys.platform != "win32",
        )

    # Wait for it to bind so the message is not a lie.
    for _ in range(60):
        if _port_in_use(args.host, port):
            break
        if child.poll() is not None:
            log.error("the server exited immediately - see %s", log_file)
            return 1
        time.sleep(0.5)

    if not _port_in_use(args.host, port):
        log.error("the server did not start listening - see %s", log_file)
        return 1

    log.info("ViralForge is running in the background at %s (pid %d)", url, child.pid)
    log.info("  log   : %s", log_file)
    log.info("  stop   : taskkill /PID %d /F", child.pid)
    log.info("You can close this window; the server keeps running.")
    _show_ui(args, url)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="clipforge",
        description="Turn long videos into vertical, subtitled short clips.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8821)
    parser.add_argument("--reload", action="store_true", help="auto-reload on code changes")
    parser.add_argument("--open", action="store_true", help="open the UI in a browser")
    parser.add_argument(
        "--window", action="store_true",
        help="open the UI in its own window instead of a browser tab, which "
             "leaves much more of the GPU free for transcription",
    )
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="with --window, cover the whole screen including the taskbar and "
             "title bar. Without it the window is maximised, which still fills "
             "the screen but keeps the close, minimise and maximise buttons",
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="start the server with no interface of any kind. The desktop "
             "launcher (ViralForge.exe) builds on this and opens the native "
             "window itself with pythonw -m clipforge.native_window.",
    )
    parser.add_argument(
        "--windowed", action="store_true", help="kept for symmetry with --fullscreen"
    )
    parser.add_argument(
        "--detach", action="store_true",
        help="run in the background so closing this window does not stop the server",
    )
    parser.add_argument("--log-level", default="info")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from .render import ffmpeg

    if not ffmpeg.available():
        log.error("ffmpeg was not found on PATH. ViralForge cannot encode without it.")
        log.error("Install FFmpeg (https://ffmpeg.org/download.html) and try again.")
        return 2
    log.info("ffmpeg: %s", ffmpeg.ffmpeg_version())

    from .engines import asr

    info = asr.device_info()
    if info.get("available"):
        log.info("speech-to-text: faster-whisper on %s (%s)",
                 info.get("resolved_device"), info.get("resolved_compute"))
    else:
        log.warning("faster-whisper is not installed; run: pip install faster-whisper")

    dist = ROOT / "web" / "dist" / "index.html"
    if not dist.exists():
        log.warning("UI bundle missing - build it with: cd web && npm install && npm run build")

    # An already-running instance wins over starting a second one. --reload is
    # the exception: it exists precisely to replace the running server.
    if not args.reload:
        existing = _running_instance(args.host, args.port)
        if existing is not None:
            log.info("ViralForge is already running at %s - opening it", existing)
            # A window is still worth opening here, otherwise asking for one
            # while the server is up silently gives you a browser tab.
            if args.window or args.open:
                _show_ui(args, existing)
            return 0

    port = _find_port(args.host, args.port)
    if port != args.port:
        log.warning("port %d is in use by something else, using %d instead",
                    args.port, port)

    url = f"http://{args.host}:{port}"

    if args.detach:
        return _run_detached(args, port, url)

    log.info("ViralForge is ready at %s", url)
    _show_ui(args, url)

    import uvicorn

    uvicorn.run(
        "clipforge.app:app",
        host=args.host,
        port=port,
        reload=args.reload,
        log_level=args.log_level,
        access_log=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
