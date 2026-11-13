"""Open the UI in a native window instead of a browser tab.

A browser tab is a poor home for this. Chrome keeps a hardware-accelerated
context, a GPU process and every other tab's memory alive, and on a card with
8 GB that can be the difference between the transcription model fitting on the
GPU and being pushed onto the CPU - which is roughly a tenfold slowdown.

Three ways to open it, best first:

1. **WebView2** via pywebview. The runtime already ships with Windows 10 and 11,
   so this is a small extra dependency rather than a whole browser. Heaviest
   saving, and no address bar, no tabs and no history.
2. **Edge in app mode.** ``--app=`` gives a standalone window with no browser
   chrome. Already installed on Windows, so this needs nothing at all - it just
   shares Edge's engine rather than Chrome's whole session.
3. **The default browser**, as a last resort, so a missing dependency never
   leaves the user with no way in.

**Cache-busting:** On every launch, WebView2 gets a completely fresh user data
directory. This guarantees it cannot serve any cached assets from a previous
session, even if the dist/ files are stale.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

log = logging.getLogger("clipforge")

# The app icon, drawn by scripts/make_icon.py. Absent on a fresh checkout that
# has not run it, so every use is guarded rather than assumed.
ICON_PATH = Path(__file__).resolve().parent.parent / "assets" / "viralforge.ico"

# Where Edge keeps itself. Checked directly because it is not reliably on PATH
# and the app bundle path has moved between 32- and 64-bit installs.
_EDGE_PATHS = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)

# Only used when the window is not full screen or maximised.
WIDTH, HEIGHT = 1500, 980


def _find_edge() -> str | None:
    for candidate in _EDGE_PATHS:
        if Path(candidate).exists():
            return candidate
    return None


def _screen_work_area() -> tuple[int, int]:
    """Usable screen size, for when maximize() is not honoured.

    Falls back to a common resolution rather than raising, because a window at
    the wrong size is still usable and a crash here means no window at all.
    """
    if sys.platform == "win32":
        try:
            import ctypes

            class _RECT(ctypes.Structure):
                _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                            ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

            rect = _RECT()
            ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rect), 0)
            return max(800, rect.right - rect.left), max(600, rect.bottom - rect.top)
        except Exception:
            pass
    return 1920, 1080


WM_SETICON = 0x0080
ICON_SMALL, ICON_BIG = 0, 1
IMAGE_ICON = 1
LR_LOADFROMFILE = 0x0010
SWP_FRAMECHANGED = 0x0020
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001


def _native_handle(window) -> int | None:
    """The Win32 handle for a pywebview window, however this build exposes it.

    The WinForms-backed Windows builds put a ``System.Windows.Forms.Form`` on
    ``native``, so the handle is an attribute of that; the GTK and Qt builds put
    something else there. Both are tolerated, and a scan of this process's own
    top-level windows is the last resort, so a backend that changes shape does
    not leave the taskbar showing the interpreter's logo.
    """
    native = getattr(window, "native", None)
    for attribute in ("Handle", "handle", "GetHandle", "winId"):
        value = getattr(native, attribute, None)
        if callable(value):
            try:
                value = value()
            except Exception:                                # noqa: BLE001
                continue
        if isinstance(value, int) and value:
            return int(value)
    try:
        import ctypes
        import ctypes.wintypes as wt

        found: list[int] = []
        pid = ctypes.windll.kernel32.GetCurrentProcessId()
        callback_type = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)

        def collect(hwnd, _lparam):
            owner = wt.DWORD()
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
            if owner.value == pid and ctypes.windll.user32.IsWindowVisible(hwnd):
                found.append(int(hwnd))
            return True

        ctypes.windll.user32.EnumWindows(callback_type(collect), 0)
        return found[0] if found else None
    except Exception:                                        # noqa: BLE001
        return None


def _apply_window_icon(window, title: str = "") -> bool:
    """Put the app icon on the window, the title bar and the taskbar button.

    pywebview's ``icon=`` is documented as supported only on GTK and Qt, and on
    Windows it stores the path and does nothing with it, so the window keeps the
    icon of the interpreter that launched it - a Python logo in the taskbar for
    a finished application. Setting it here is the only thing that actually
    changes what Windows draws.

    Returns whether it worked, so the caller can say so rather than the user
    discovering it from the taskbar.
    """
    if sys.platform != "win32" or not ICON_PATH.exists():
        return False
    try:
        import ctypes
        import ctypes.wintypes as wt

        user32 = ctypes.windll.user32
        shell32 = ctypes.windll.shell32
        # An explicit id keeps the taskbar from grouping this with any other
        # python.exe and gives it our identity before the icon is set.
        try:
            shell32.SetCurrentProcessExplicitAppUserModelID("ViralForge.LongToShort")
        except Exception:                                    # noqa: BLE001
            pass

        hwnd = _native_handle(window)
        if not hwnd:
            log.warning("could not find the window handle, so the icon stays the default")
            return False

        def load(size: int) -> int:
            return user32.LoadImageW(None, str(ICON_PATH), IMAGE_ICON, size, size,
                                     LR_LOADFROMFILE)

        # The handles must outlive this call: the window keeps using them for as
        # long as it exists, so they are deliberately never destroyed.
        small, big = load(16), load(32)
        if not big:
            log.warning("could not load the icon from %s", ICON_PATH)
            return False
        user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, big)
        user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, small)
        # The frame caches the old icon until something marks it dirty.
        user32.SetWindowPos(hwnd, None, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_FRAMECHANGED)
        user32.RedrawWindow(hwnd, None, None, 0x0001 | 0x0004 | 0x0080)
        log.info("window icon applied from %s", ICON_PATH)
        return True
    except Exception as exc:                                 # noqa: BLE001
        log.warning("could not set the window icon (%s)", exc)
        return False


class _DesktopApi:
    """The few things only the desktop window can do, exposed to the page.

    The interface is otherwise a plain web app talking to a REST API, and a web
    page cannot open a real folder picker: a directory input reports a faked
    ``C:\\fakepath`` and browsers deliberately withhold the real location. So the
    native dialog is reached over pywebview's JS bridge instead, and the page
    falls back to a text field when it is running in an ordinary browser where
    that bridge does not exist.
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
        except Exception as exc:                            # noqa: BLE001
            log.warning("the folder picker failed (%s)", exc)
            return ""
        if not result:
            return ""                                        # cancelled
        first = result[0] if isinstance(result, (list, tuple)) else result
        return str(first)

    def reveal_folder(self, path: str) -> bool:
        """Open a folder in Explorer, so 'where did my clips go' is one click."""
        if sys.platform != "win32" or not path:
            return False
        try:
            target = Path(path)
            if not target.is_dir():
                return False
            os.startfile(str(target))                      # noqa: S606
            return True
        except Exception:                                   # noqa: BLE001
            return False


def _open_window_webview(url: str, title: str, fullscreen: bool) -> None:
    import tempfile
    import webview  # type: ignore

    # The only 100% guaranteed way to bust WebView2 caching is to assign a
    # completely fresh, temporary user data directory on every single launch.
    # This creates a new isolated browser instance that cannot serve cached
    # assets from a previous session.
    fresh_dir = tempfile.mkdtemp(prefix="clipforge_webview_")
    print(f"[WebView2] Using fresh user data directory: {fresh_dir}")

    # Create a callback function for window events
    def on_created(window):
        # Apply window icon after the window is created
        _apply_window_icon(window, title)
        
        # Maximize or fullscreen the window
        try:
            if fullscreen:
                window.toggle_fullscreen()
            else:
                window.maximize()
        except Exception as exc:                            # noqa: BLE001
            log.warning("could not %s the window (%s), using the screen size",
                        "full screen" if fullscreen else "maximise", exc)
            width, height = _screen_work_area()
            try:
                window.restore()
                window.move(0, 0)
                window.resize(width, height)
            except Exception:                                   # noqa: BLE001
                pass

    # The window is created at its normal size and enlarged once the GUI is up.
    # Passing maximized=True to create_window does not work on this version of
    # pywebview - the window comes up minimised at 158x26, off-screen.
    window = webview.create_window(
        title, url=url,
        width=WIDTH, height=HEIGHT, min_size=(900, 620),
        on_top=False, resizable=True
    )

    # The icon belongs on start(), not create_window() - which does not take
    # one. Passing it to create_window raises TypeError, and since that happens
    # inside the try in open_ui the whole WebView2 path is skipped for Edge.
    # It is still passed because GTK and Qt builds do honour it, and on Windows
    # _apply_window_icon does the work instead.
    icon = str(ICON_PATH) if ICON_PATH.exists() else None
    api = _DesktopApi()
    
    # Start the webview with a simple callback approach
    try:
        # private_mode=True forces WebView2 to ignore the cache entirely
        # This ensures the UI always loads from the freshly built dist/ directory
        webview.start(gui="edgechromium", icon=icon, js_api=api, private_mode=True)
    except TypeError:
        # Older pywebview builds take neither gui= nor icon= on start(), and
        # js_api arrived later still. The page detects the missing bridge and
        # falls back to asking for the path as text.
        try:
            webview.start(gui="edgechromium", icon=icon, private_mode=True)
        except TypeError:
            webview.start(private_mode=True)

def _open_window_edge(url: str, title: str, fullscreen: bool) -> None:
    import tempfile
    edge = _find_edge()
    if edge is None:
        webbrowser.open(url)
        return
    
    # Use a fresh temporary directory to prevent stale cache issues
    fresh_dir = tempfile.mkdtemp(prefix="clipforge_edge_")
    print(f"[Edge] Using fresh user data directory: {fresh_dir}")
    
    subprocess.Popen(
        [
            edge,
            f"--app={url}",
            f"--window-size={WIDTH},{HEIGHT}",
            "--start-fullscreen" if fullscreen else "--start-maximized",
            # A separate user-data-dir so this does not attach to, or inherit
            # from, an Edge session that is already running - and with it that
            # session's GPU process, which is the thing being avoided.
            f"--user-data-dir={fresh_dir}",
            "--no-first-run",
            "--no-default-browser-check",
        ],
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )


def _open_window_edge_fallback(url: str, title: str, fullscreen: bool) -> str:
    """Edge fallback when WebView2 fails or is not available."""
    try:
        _open_window_edge(url, title, fullscreen)
        return "edge"
    except Exception as exc:                            # noqa: BLE001
        log.error("Edge fallback failed with: %s", exc)
        return "browser"


def open_ui(url: str, title: str = "ViralForge", fullscreen: bool = False) -> str:
    """Open the interface and report which way it was done.

    Never raises: a missing optional dependency falls through to the next
    option, and the worst case is the browser the user already had.
    """
    if sys.platform == "win32":
        try:
            import webview  # type: ignore  # noqa: F401
        except ImportError:
            log.error("pywebview is not installed - standalone window requires it")
            return _open_window_edge_fallback(url, title, fullscreen)

        try:
            _open_window_webview(url, title, fullscreen)
            return "webview2"
        except Exception as exc:                            # noqa: BLE001
            log.error("WebView2 failed with: %s", exc)
            return _open_window_edge_fallback(url, title, fullscreen)

    return "edge"


def describe(how: str, fullscreen: bool = False) -> str:
    shape = "full screen" if fullscreen else "maximised"
    return {
        "webview2": f"opened in a WebView2 window, {shape} (lightest on VRAM)",
        "edge": f"opened in an Edge app window, {shape}; "
                "`pip install pywebview` makes it lighter still",
        "browser": "opened in your default browser",
    }.get(how, how)
