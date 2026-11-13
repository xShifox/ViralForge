"""The interface-opening path: a maximised window by default, a tab only if asked.

These exist because the failure is silent. A name that is only defined inside
main() raises NameError the moment a window is requested, and the launcher then
exits having opened nothing at all - no window, no tab, no message.
"""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


class _Args:
    """Just the flags the interface-opening helper looks at."""

    def __init__(self, window: bool = False, open_: bool = False,
                 fullscreen: bool = False) -> None:
        self.window = window
        self.open = open_
        self.fullscreen = fullscreen


def test_nothing_requested_opens_nothing(monkeypatch):
    from clipforge import __main__ as entry

    opened: list[str] = []
    monkeypatch.setattr(entry.webbrowser, "open", opened.append)

    entry._show_ui(_Args(window=False, open_=False), "http://x")
    assert opened == []


def test_open_gives_exactly_one_browser(monkeypatch):
    from clipforge import __main__ as entry

    opened: list[str] = []
    monkeypatch.setattr(entry.webbrowser, "open", opened.append)

    entry._show_ui(_Args(window=False, open_=True), "http://x")
    assert opened == ["http://x"]


def test_window_gives_a_window_and_not_also_a_tab(monkeypatch):
    """The original double-tab bug, in its new shape.

    --window and --open can both be present, and asking for both must not open
    two things.
    """
    from clipforge import __main__ as entry
    from clipforge import window as window_mod

    opened: list[str] = []
    windows: list[str] = []
    monkeypatch.setattr(entry.webbrowser, "open", opened.append)
    monkeypatch.setattr(window_mod, "open_ui",
                        lambda url, title="ViralForge", fullscreen=False:
                            windows.append(url) or "webview2")

    entry._show_ui(_Args(window=True, open_=True), "http://x")
    assert windows == ["http://x"]
    assert opened == [], "a window and a browser tab were both opened"


def test_a_window_is_still_opened_when_the_server_is_already_up(monkeypatch):
    """Asking for a window must not quietly give a tab instead.

    The already-running path is the common one - the server outlives the window
    by design - so getting this wrong means the window never appears at all.
    """
    from clipforge import __main__ as entry

    seen: list[str] = []

    class Args:
        host, port, log_level = "127.0.0.1", 8756, "info"
        detach = reload = open = window = fullscreen = False

    monkeypatch.setattr(entry, "_running_instance", lambda host, port: "http://127.0.0.1:8756")
    monkeypatch.setattr(entry, "_show_ui", lambda args, url: seen.append(url))

    assert entry.main(["--window"]) == 0
    assert seen == ["http://127.0.0.1:8756"], (
        "a window was requested while the server was up, and nothing was opened")


def test_the_window_helper_never_raises(monkeypatch):
    """A missing optional dependency degrades instead of exploding.

    pywebview is optional, so its absence is a normal outcome; the user must
    still get an interface, worst case the browser they already had.
    """
    from clipforge import window as window_mod

    calls: list[str] = []
    monkeypatch.setattr(window_mod.sys, "platform", "linux")
    monkeypatch.setattr(window_mod, "_open_window_edge",
                        lambda url, title, fullscreen: calls.append("edge"))

    assert window_mod.open_ui("http://x") == "edge"
    assert calls == ["edge"]

    def boom(url, title, fullscreen):
        raise OSError("no display")

    monkeypatch.setattr(window_mod, "_open_window_edge", boom)
    monkeypatch.setattr(window_mod.webbrowser, "open", lambda url: calls.append("browser"))
    assert window_mod.open_ui("http://x") == "browser"
    assert calls[-1] == "browser"


def test_the_window_opens_maximised_not_full_screen(monkeypatch):
    """Maximised, not full screen.

    Full screen also covers the taskbar and the title bar, which removes the
    close, minimise and maximise buttons and leaves only Alt+F4 to get out. That
    was tried, and it made the app hard to leave.
    """
    from clipforge import window as window_mod

    seen: list[bool] = []
    monkeypatch.setattr(window_mod.sys, "platform", "linux")
    monkeypatch.setattr(window_mod, "_open_window_edge",
                        lambda url, title, fullscreen: seen.append(fullscreen))

    window_mod.open_ui("http://x")
    assert seen == [False], "the window did not open maximised by default"


def test_full_screen_is_opt_in(monkeypatch):
    from clipforge import window as window_mod

    seen: list[bool] = []
    monkeypatch.setattr(window_mod.sys, "platform", "linux")
    monkeypatch.setattr(window_mod, "_open_window_edge",
                        lambda url, title, fullscreen: seen.append(fullscreen))

    window_mod.open_ui("http://x", fullscreen=True)
    assert seen == [True]


def test_the_flag_reaches_both_backends(monkeypatch):
    """Both the WebView2 and the Edge path have to honour the choice.

    They express the same decision differently - maximized is the negation of
    fullscreen - so a change to one and not the other is easy to make.
    """
    from clipforge import __main__ as entry
    from clipforge import window as window_mod

    source = (ROOT / "clipforge" / "window.py").read_text(encoding="utf-8")
    assert '"--start-fullscreen" if fullscreen else "--start-maximized"' in source
    # The WebView2 window is enlarged after the GUI is up, not at creation:
    # passing maximized=True to create_window leaves it minimised off-screen.
    assert "window.maximize()" in source
    assert "window.toggle_fullscreen()" in source

    # And the launcher has to pass the user's choice through.
    seen: list[bool] = []
    monkeypatch.setattr(window_mod, "open_ui",
                        lambda url, title="ViralForge", fullscreen=False:
                            seen.append(fullscreen) or "webview2")
    entry._show_ui(_Args(window=True, fullscreen=False), "http://x")
    entry._show_ui(_Args(window=True, fullscreen=True), "http://x")
    assert seen == [False, True]


def test_the_window_carries_the_app_icon(monkeypatch):
    """A generic blank icon is a missed detail on the taskbar and in Alt+Tab.

    This used to assert only that the icon reached ``webview.start()``, and it
    passed while the taskbar showed a Python logo for months: pywebview documents
    that argument as GTK/QT only, so on Windows it stored the path and did
    nothing. The test checked the call, not the outcome, which is the one thing
    worth checking.
    """
    from clipforge import window as window_mod

    source = (ROOT / "clipforge" / "window.py").read_text(encoding="utf-8")
    # Guarded, because a checkout that has not run scripts/make_icon.py must
    # still start - so the path is checked, not assumed.
    assert "ICON_PATH.exists()" in source, "the icon is used without checking it exists"
    # The argument is still passed, because GTK and Qt builds do honour it.
    assert "webview.start(fill_the_screen, gui=\"edgechromium\", icon=icon)" in source
    # But on Windows it has to be set on the window handle, or nothing happens.
    assert "_apply_window_icon(window, title)" in source, (
        "the icon is only passed to pywebview, which ignores it on Windows")
    assert "WM_SETICON" in source, "the window handle is never given the icon"
    assert "ICON_PATH" in source and "assets" in source


def test_the_icon_is_applied_to_the_window_handle(monkeypatch):
    """The icon has to be set on the HWND, so this checks the calls it makes."""
    from clipforge import window as window_mod

    sent: list[tuple[int, int]] = []

    class _User32:
        def __init__(self):
            self.LoadImageW = self._load
            self.SendMessageW = self._send
            self.SetWindowPos = lambda *a: 0
            self.RedrawWindow = lambda *a: 0

        @staticmethod
        def _load(_inst, path, kind, cx, cy, flags):
            return 0xABCD if cx >= 32 else 0x1234

        @staticmethod
        def _send(hwnd, msg, wparam, lparam):
            sent.append((msg, wparam))
            return 1

    class _Shell32:
        @staticmethod
        def SetCurrentProcessExplicitAppUserModelID(_app_id):
            return 0

    class _Windll:
        user32 = _User32()
        shell32 = _Shell32()

    monkeypatch.setattr(window_mod.sys, "platform", "win32")
    monkeypatch.setattr("ctypes.windll", _Windll)
    monkeypatch.setattr(window_mod, "_native_handle", lambda _w: 4321)

    class _FakeWindow:
        native = None

    assert window_mod._apply_window_icon(_FakeWindow()) is True
    messages = {msg for msg, _wparam in sent}
    assert window_mod.WM_SETICON in messages, "the window was never told about the icon"
    # Both the title bar and the taskbar button, or the taskbar keeps the default.
    wparams = {wparam for msg, wparam in sent if msg == window_mod.WM_SETICON}
    assert wparams == {window_mod.ICON_BIG, window_mod.ICON_SMALL}, wparams


def test_a_missing_icon_file_is_not_fatal(monkeypatch, tmp_path):
    """A checkout without the built icon must still open a window."""
    from clipforge import window as window_mod

    monkeypatch.setattr(window_mod.sys, "platform", "win32")
    monkeypatch.setattr(window_mod, "ICON_PATH", tmp_path / "absent.ico")
    assert window_mod._apply_window_icon(object()) is False


def test_it_declines_rather_than_raising_off_windows(monkeypatch):
    from clipforge import window as window_mod

    monkeypatch.setattr(window_mod.sys, "platform", "linux")
    assert window_mod._apply_window_icon(object()) is False


def test_the_native_handle_is_read_from_whichever_shape_the_backend_uses():
    """The Windows backends put different things on ``native``; all are tolerated."""
    from clipforge import window as window_mod

    class _WithHandle:
        native = type("Form", (), {"Handle": 4242})()

    class _WithCall:
        native = type("Qt", (), {"winId": staticmethod(lambda: 7777)})()

    class _WithNothing:
        native = object()

    assert window_mod._native_handle(_WithHandle()) == 4242
    assert window_mod._native_handle(_WithCall()) == 7777
    # Nothing usable, and no window on this machine to fall back to.
    assert window_mod._native_handle(_WithNothing()) in (None, 0)


def test_the_edge_fallback_is_a_standalone_window():
    """--app= is what makes it a window; without it, it is just another tab.

    The fallback matters because it is the path taken when pywebview is not
    installed, and it is still far lighter than a full browser session.
    """
    source = (ROOT / "clipforge" / "window.py").read_text(encoding="utf-8")
    assert '"--app=' in source, "the Edge fallback lost its app-window flag"
    # A shared profile would inherit an already-running Edge session and its GPU
    # process, which is the very thing being avoided.
    assert "--user-data-dir" in source


def test_the_detached_server_gets_no_console():
    """No console for the server, so nothing flashes on the taskbar.

    CREATE_NO_WINDOW rather than DETACHED_PROCESS: the detached one still leaves
    a console host process behind, and that is the black box that appears and
    disappears on every launch.
    """
    source = (ROOT / "clipforge" / "__main__.py").read_text(encoding="utf-8")
    # Comments are stripped first, because the file names DETACHED_PROCESS in a
    # comment explaining why it is not used - matching the word would test the
    # prose rather than the flag.
    code = "\n".join(l for l in source.splitlines() if not l.lstrip().startswith("#"))
    assert "0x08000000" in code, "CREATE_NO_WINDOW is not being passed to the server"
    assert "0x00000008" not in code, "the old DETACHED_PROCESS flag is back"


def test_there_is_a_launcher_with_no_console_at_all():
    """start.bat always has a console, so a console-free entry point is needed.

    WScript.Shell's Run with window style 0 creates no console even briefly,
    which is the only way to stop the flash on a double-click.
    """
    vbs = ROOT / "ViralForge.vbs"
    assert vbs.exists(), "there is no console-free launcher to double-click"
    text = vbs.read_text(encoding="utf-8", errors="replace")
    assert "shell.Run command, 0, False" in text, (
        "the VBS launcher is not passing window style 0, so it will flash a console")
    assert "start.bat" in text


@pytest.mark.parametrize("how", ["webview2", "edge", "browser", "something-else"])
def test_every_window_outcome_is_described(how):
    """The launcher says which kind of window appeared.

    Otherwise "no tab opened" and "a window opened" are indistinguishable from
    the console, and the user has no way to tell what happened.
    """
    from clipforge.window import describe

    assert describe(how)
    if how in ("webview2", "edge"):
        # The message says which shape it opened in, so the console is not the
        # only place that knows whether the taskbar is covered.
        assert describe(how, fullscreen=True) != describe(how, fullscreen=False)
