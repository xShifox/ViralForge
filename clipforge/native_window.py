#!/usr/bin/env python
"""Open the ViralForge UI in a native pywebview window.

Run with pythonw (no console):

    pythonw.exe -m clipforge.native_window            # http://127.0.0.1:8821
    pythonw.exe -m clipforge.native_window --url X    # a different address

pythonw has no console, so everything is logged to ``data/launcher.log``
alongside the desktop launcher's own output.

The page needs the JS bridge for the folder picker in the export dialog:
a directory input reports a faked ``C:\\fakepath`` and browsers withhold the
real location on purpose, so the only way to choose a real folder is pywebview's
``js_api``. It is wired onto ``create_window``, not ``start`` - ``start`` has no
such parameter and passing it there silently drops the bridge.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "data"
ICON_PATH = ROOT / "assets" / "viralforge.ico"
DEFAULT_URL = "http://127.0.0.1:8821"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename=str(LOG_DIR / "launcher.log"),
    encoding="utf-8",
)
log = logging.getLogger("clipforge.native_window")


def _app_id() -> None:
    """Give the window its own taskbar identity instead of pythonw.exe's."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            "ViralForge.Desktop.App")
    except Exception as exc:  # noqa: BLE001
        log.warning("could not set the AppUserModelID: %s", exc)


def _enum_windows_by_title(title: str) -> list[int]:
    """Visible top-level windows of this process whose title contains ``title``."""
    if sys.platform != "win32":
        return []
    try:
        import ctypes
        import ctypes.wintypes as wt

        user32 = ctypes.windll.user32
        pid = ctypes.windll.kernel32.GetCurrentProcessId()
        found: list[int] = []
        callback_type = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

        def collect(hwnd, _lparam):
            owner = wt.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value != pid or not user32.IsWindowVisible(hwnd):
                return True
            buf = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, buf, 255)
            if title.lower() in buf.value.lower():
                found.append(int(hwnd))
            return True

        user32.EnumWindows(callback_type(collect), 0)
        return found
    except Exception as exc:  # noqa: BLE001
        log.warning("could not enumerate windows: %s", exc)
        return []


def _apply_window_icon(window) -> None:
    """Put the gold icon on the title bar and the taskbar button."""
    if sys.platform != "win32" or not ICON_PATH.exists():
        return
    try:
        import ctypes

        user32 = ctypes.windll.user32
        WM_SETICON, ICON_SMALL, ICON_BIG, IMAGE_ICON, LR_LOADFROMFILE = 0x0080, 0, 1, 1, 0x0010

        hwnd = None
        native = getattr(window, "native", None)
        for attribute in ("Handle", "handle", "GetHandle", "winId"):
            value = getattr(native, attribute, None)
            if callable(value):
                try:
                    value = value()
                except Exception:  # noqa: BLE001
                    continue
            if isinstance(value, int) and value:
                hwnd = int(value)
                break
        if not hwnd:
            # Qt's winId can be unavailable this early; enumerating our own
            # process's visible windows is the reliable fallback.
            matches = _enum_windows_by_title("ViralForge")
            hwnd = matches[0] if matches else None
        if not hwnd:
            log.warning("could not find the window handle; the icon stays the default")
            return

        def load(size: int) -> int:
            return user32.LoadImageW(None, str(ICON_PATH), IMAGE_ICON, size, size,
                                     LR_LOADFROMFILE)

        small, big = load(16), load(32)
        if not big:
            log.warning("could not load the icon from %s", ICON_PATH)
            return
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)
        user32.SetWindowPos(hwnd, None, 0, 0, 0, 0,
                            0x0001 | 0x0002 | 0x0020)  # SWP_NOSIZE|SWP_NOMOVE|SWP_FRAMECHANGED
        log.info("window icon applied from %s", ICON_PATH)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not set the window icon: %s", exc)


class _DesktopApi:
    """The two things the page cannot do from a normal browser.

    The interface is otherwise a plain web app talking to the REST API; only
    these two need the native window. Everything else stays out of reach.
    """

    def pick_folder(self, start: str = "") -> str:
        """Open a folder chooser. Returns the chosen path, or "" if cancelled."""
        try:
            import webview  # type: ignore

            windows = webview.windows
            if not windows:
                return ""
            result = windows[0].create_file_dialog(
                webview.FOLDER_DIALOG,
                directory=start or None,
                allow_multiple=False,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("the folder picker failed: %s", exc)
            return ""
        if not result:
            return ""
        first = result[0] if isinstance(result, (list, tuple)) else result
        return str(first)

    def reveal_folder(self, path: str) -> bool:
        """Open a folder in Explorer."""
        import os

        try:
            target = Path(path)
            return bool(path) and target.is_dir() and os.startfile(str(target)) is None
        except Exception:  # noqa: BLE001
            return False

    def pick_file(self, start: str = "", pattern: str = "*.mp3;*.wav;*.m4a;*.aac;*.flac") -> str:
        """Open a single audio file chooser. Returns the chosen path, or "" if cancelled."""
        try:
            import webview  # type: ignore

            windows = webview.windows
            if not windows:
                return ""
            patterns = [p.strip() for p in pattern.split(";") if p.strip()]
            file_types = (f"Audio files ({';'.join(patterns)})",) if patterns else ()
            result = windows[0].create_file_dialog(
                webview.OPEN_DIALOG,
                directory=start or None,
                allow_multiple=False,
                file_types=file_types,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("the file picker failed: %s", exc)
            return ""
        if not result:
            return ""
        first = result[0] if isinstance(result, (list, tuple)) else result
        return str(first)


def main() -> int:
    parser = argparse.ArgumentParser(description="ViralForge native window")
    parser.add_argument("--url", default=DEFAULT_URL, help="URL to load")
    parser.add_argument("--fullscreen", action="store_true")
    args = parser.parse_args()

    log.info("native window starting for %s", args.url)
    _app_id()

    try:
        import webview  # type: ignore
    except Exception as exc:  # noqa: BLE001
        log.exception("pywebview is not importable: %s", exc)
        _fatal(f"pywebview failed to import: {exc}")
        return 1

    api = _DesktopApi()

    # The window is created at a normal size and enlarged once the GUI is up;
    # maximized=True in create_window brings the window up minimised off-screen.
    window = webview.create_window(
        "ViralForge", args.url,
        js_api=api,
        width=1500, height=980,
        min_size=(900, 620),
        background_color="#0b0b0f",
    )

    def fill_the_screen() -> None:
        try:
            if args.fullscreen:
                window.toggle_fullscreen()
            else:
                window.maximize()
        except Exception as exc:  # noqa: BLE001
            log.warning("could not maximise the window: %s", exc)
        _apply_window_icon(window)

    icon = str(ICON_PATH) if ICON_PATH.exists() else None
    try:
        webview.start(fill_the_screen, gui="qt", icon=icon)
        log.info("native window closed")
        return 0
    except Exception as exc:  # noqa: BLE001
        log.exception("the Qt window failed (%s); trying the WebView2 backend", exc)
        try:
            webview.start(fill_the_screen, gui="edgechromium", icon=icon)
            log.info("native window closed")
            return 0
        except Exception as exc2:  # noqa: BLE001
            log.exception("the native window could not be created: %s", exc2)
            _fatal(str(exc2))
            return 1


def _fatal(message: str) -> None:
    log.error("fatal: %s", message)
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.user32.MessageBoxW(
                0,
                f"ViralForge could not open its window.\n\n{message}\n\n"
                f"Details are in {LOG_DIR / 'launcher.log'}",
                "ViralForge could not start", 0x10)
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
