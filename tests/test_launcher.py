"""Starting the app must not start a second app.

Two separate faults, both easy to reintroduce because neither shows up in a unit
test of the functions involved:

* the detached launcher passed --open to the child *and* opened a browser
  itself, so every launch opened two tabs;
* a busy port was read as "use the next one", which quietly started a rival
  server sharing the same job directory.

These use the real port-scanning and probing code, with a live server, because
the behaviour is about sockets and processes rather than return values.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from clipforge.__main__ import _find_port, _port_in_use, _running_instance

ROOT = Path(__file__).resolve().parent.parent
HOST = "127.0.0.1"


def _listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.4)
        return sock.connect_ex((HOST, port)) == 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


def test_no_instance_is_reported_for_a_free_port():
    port = _free_port()
    assert not _port_in_use(HOST, port)
    assert _running_instance(HOST, port) is None


def _serve(payload: bytes) -> tuple[int, object]:
    """A real HTTP server on a free port, answering every request with payload."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:                        # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args) -> None:
            pass

    server = HTTPServer((HOST, 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_port, server


def test_a_foreign_service_on_the_port_is_not_mistaken_for_clipforge():
    """A busy port is not a running ViralForge.

    The port fallback has to stay available for unrelated programs, so the probe
    must refuse anything that does not answer like ViralForge - including one that
    returns a 200 with {"ok": true}, which is the shape most likely to fool it.
    """
    port, server = _serve(b'{"ok": true}')
    try:
        assert _running_instance(HOST, port, timeout=2.0) is None
    finally:
        server.shutdown()
        server.server_close()


def test_something_listening_but_not_answering_is_not_an_instance():
    port = _free_port()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, port))
        sock.listen(1)
        assert _running_instance(HOST, port, timeout=0.5) is None


def test_a_clipforge_health_response_is_recognised():
    """The positive case, so the test above is not just rejecting everything."""
    payload = json.dumps({
        "ok": True, "ffmpeg": True, "ffmpeg_version": "ffmpeg version 9.0.1",
        "data_dir": "C:/data", "asr": {}, "llm": {}, "platforms": {},
        "limits": {}, "queue": {}, "encoder": {},
    }).encode()
    port, server = _serve(payload)
    try:
        assert _running_instance(HOST, port, timeout=2.0) == f"http://{HOST}:{port}"
    finally:
        server.shutdown()
        server.server_close()


@pytest.mark.slow
def test_a_second_launch_reuses_the_running_server(tmp_path):
    port = _free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "clipforge", "--host", HOST, "--port", str(port),
         "--log-level", "warning"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 120
        while time.time() < deadline and not _listening(port):
            if server.poll() is not None:
                pytest.fail("the test server exited during startup")
            time.sleep(0.5)
        assert _listening(port), "the test server never bound its port"

        assert _running_instance(HOST, port) == f"http://{HOST}:{port}"

        # Launching again must not stand up a rival on the next port.
        second = subprocess.run(
            [sys.executable, "-m", "clipforge", "--host", HOST, "--port", str(port),
             "--log-level", "warning"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=180,
        )
        assert second.returncode == 0, second.stderr[-300:]
        assert server.poll() is None, "the second launch replaced the first server"
        assert not _listening(port + 1), "a duplicate server was started"
    finally:
        server.terminate()
        try:
            server.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server.kill()


def test_a_launch_opens_the_interface_exactly_once():
    """One launch, one tab or window.

    Passing --open to the detached child *and* opening a browser in the parent
    opened two tabs every single time, because start.bat runs --open --detach.
    The child is now told nothing about opening anything, and the parent does it
    once, through a single helper.
    """
    source = (ROOT / "clipforge" / "__main__.py").read_text(encoding="utf-8")

    # The detached child command must not carry --open.
    detached = source.split("def _run_detached", 1)[1].split("\ndef main", 1)[0]
    assert '"--open"' not in detached, (
        "the detached child is still passed --open, so it opens its own tab")

    # And the browser is opened from exactly one place in the whole file.
    calls = source.count("webbrowser.open(")
    assert calls == 1, f"expected one webbrowser.open call, found {calls}"
