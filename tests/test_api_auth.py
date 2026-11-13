﻿"""The API must not be open to anything that can open a socket.

The product binds to loopback, which makes "any local process may use it" the
normal case and arguably fine for a single-user desktop app. Two things stop that
being fine: the API can enumerate any directory on any drive, rewrite global
settings and delete job folders; and ``--host`` is one argument away from putting
all of that on a network.

So the API requires a token generated per installation. These tests are the only
ones in the suite that run with authentication *enforced* - everything else turns
it off, which is why the conftest marker exists.

They also cover the two things that token is useless without: that a correct
token cannot be laundered into an arbitrary file read, and that the browser is
actually given one.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.authtoken

from clipforge.app import AUTH_DISABLE_ENV, create_app  # noqa: E402
from clipforge.core import authtoken                  # noqa: E402


@pytest.fixture(autouse=True)
def _no_bypass(monkeypatch):
    """Authentication is the subject here, so the suite-wide bypass must be off."""
    monkeypatch.delenv(AUTH_DISABLE_ENV, raising=False)
    monkeypatch.delenv("CLIPFORGE_NO_TOKEN", raising=False)
    authtoken.invalidate()
    yield
    authtoken.invalidate()


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_app())


@pytest.fixture()
def token() -> str:
    return authtoken.load_or_create(Path(r"<TOOL_ROOT>\data"))


# ------------------------------------------------------------------ enforcement

def test_an_api_call_without_a_token_is_refused(client):
    response = client.get("/api/system/health")
    assert response.status_code == 401
    assert "token" in response.json()["detail"].lower()


def test_a_wrong_token_is_refused(client, token):
    response = client.get("/api/system/health",
                          headers={"Authorization": "Bearer not-the-token"})
    assert response.status_code == 401


def test_an_empty_bearer_is_refused(client, token):
    response = client.get("/api/system/health",
                          headers={"Authorization": "Bearer "})
    assert response.status_code == 401


def test_the_right_token_is_accepted(client, token):
    response = client.get("/api/system/health",
                          headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200, response.text


def test_the_token_may_also_arrive_as_a_cookie(client, token):
    """A plain page load cannot set a header, so the cookie path must work."""
    client.cookies.set("clipforge_token", token)
    response = client.get("/api/jobs")
    assert response.status_code == 200, response.text


def test_a_token_of_the_wrong_length_is_refused_without_comparing(client):
    """Length is not a secret, but an empty value must never be treated as a match."""
    assert authtoken.matches("", "anything") is False
    assert authtoken.matches("anything", "") is False
    assert authtoken.matches("ab", "abc") is False


def test_the_version_root_stays_open(client):
    """It is how a human finds out whether anything is running, and it is inert."""
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json()["name"] == "ViralForge"


def test_cors_preflight_is_not_blocked(client):
    """The browser sends OPTIONS before the real request and cannot sign it."""
    response = client.options("/api/system/settings",
                              headers={"Origin": "http://localhost:5173",
                                       "Access-Control-Request-Method": "PUT"})
    assert response.status_code != 401


def test_mutating_routes_are_covered_too(client):
    """Health is a read. These are the ones that change state on disk."""
    assert client.put("/api/system/settings", json={"max_clips": 1}).status_code == 401
    assert client.delete("/api/jobs/whatever").status_code == 401
    assert client.get("/api/system/folders").status_code == 401
    assert client.post("/api/calibration", json={}).status_code in (401, 404, 405)


# ---------------------------------------------------------------- token itself

def test_the_token_is_stable_across_calls(tmp_path):
    first = authtoken.load_or_create(tmp_path)
    authtoken.invalidate()
    second = authtoken.load_or_create(tmp_path)
    assert first == second, "a second call regenerated the token"
    assert authtoken.token_path(tmp_path).is_file()


def test_a_token_file_written_by_hand_is_used_as_is(tmp_path):
    """Somebody who needs to script against the API should not have to guess."""
    authtoken.token_path(tmp_path).write_text("a-token-i-chose", encoding="utf-8")
    authtoken.invalidate()
    assert authtoken.load_or_create(tmp_path) == "a-token-i-chose"


def test_a_token_file_that_is_not_text_is_replaced(tmp_path):
    path = authtoken.token_path(tmp_path)
    path.write_bytes(b"\xff\xfe\x00binary junk")
    authtoken.invalidate()
    token = authtoken.load_or_create(tmp_path)
    assert token and token != "\xff\xfe\x00binary junk"
    assert authtoken.load_or_create(tmp_path) == token, "the replacement must persist"


def test_regenerate_replaces_the_token(tmp_path):
    first = authtoken.load_or_create(tmp_path)
    second = authtoken.load_or_create(tmp_path, regenerate=True)
    assert first != second
    assert authtoken.load_or_create(tmp_path) == second


def test_the_token_is_long_enough_to_be_a_secret():
    import secrets as _s
    token = _s.token_urlsafe(32)
    assert len(token) >= 43, f"only {len(token)} characters"


# ------------------------------------------------------- the browser gets one

def test_the_page_hands_the_browser_a_token(client, token):
    """The cookie is set on the SPA response, which is the only response the
    browser is guaranteed to receive before it makes an API call."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("the interface is not built")
    response = client.get("/")
    assert response.status_code == 200
    cookies = response.headers.get_list("set-cookie")
    joined = " ".join(cookies)
    assert "clipforge_token=" in joined, joined
    # HttpOnly so a script cannot read it, Strict so a cross-site request that
    # looks like DNS rebinding does not carry it.
    assert "httponly" in joined.lower()
    assert "samesite=strict" in joined.lower().replace("samesite=strict", "samesite=strict")


def test_the_page_is_reachable_without_a_token(client):
    """Circular otherwise: the page is where the token is given."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("the interface is not built")
    assert client.get("/").status_code == 200


def test_a_static_asset_is_reachable_without_a_token(client):
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("the interface is not built")
    assert client.get("/index.html").status_code == 200


# ---------------------------------------------------------- path traversal

#: Canary paths that must stay unreachable. These deliberately point at real
#: files that sit outside ``web/dist``: an earlier version of this test aimed at
#: ``Windows/win.ini``, which on a machine whose project lives on D: resolves to
#: a path that does not exist - so the fallback masked the vulnerability and the
#: test passed for the wrong reason even with the check removed.
CANARIES = [
    "/%2e%2e/package.json",                 # web/dist -> web/package.json
    "/%2e%2e/%2e%2e/requirements.txt",      # -> repo root
    "/..%2f..%2frequirements.txt",
    "/assets/../../../requirements.txt",  # assets -> dist -> web -> repo root
]


@pytest.mark.parametrize("attack", CANARIES)
def test_the_spa_route_cannot_read_outside_web_dist(client, attack):
    """The catch-all resolved nothing and checked nothing, so this was an
    arbitrary file read for any caller that could reach the port.

    Only the percent-encoded form reaches the route as ``..`` - a literal
    ``/../../`` is collapsed by the path normaliser before routing, which is
    why this can sit unnoticed and why the encoded form is the one that matters.
    """
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("the interface is not built")
    target = attack.replace("%2e%2e", "..").replace("%2f", "/").lstrip("/")
    assert (ROOT / "web" / "dist" / target).resolve().is_file(), \
        f"canary {attack} no longer points at a real file; pick another"
    # Bytes, not text: read_text normalises CRLF to LF while the HTTP body keeps
    # the file's own line endings, so a text comparison fails on line endings
    # alone and tells you nothing about traversal.
    shell = (ROOT / "web" / "dist" / "index.html").read_bytes()
    response = client.get(attack)
    assert response.status_code == 200
    # A traversal must land on the SPA fallback and nowhere else. Comparing
    # against the shell in full is exact: a leaked requirements.txt cannot be
    # byte-identical to index.html, so this cannot pass by luck.
    assert response.content == shell, f"{attack} served something other than the shell"


def test_the_spa_route_still_serves_a_real_asset(client):
    """The containment check must not have broken ordinary serving."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("the interface is not built")
    response = client.get("/index.html")
    assert response.status_code == 200
    assert "<div id=" in response.text or "<script" in response.text


def test_the_spa_route_serves_an_unknown_route_as_the_shell(client):
    """Client-side routing depends on this."""
    if not (ROOT / "web" / "dist" / "index.html").is_file():
        pytest.skip("the interface is not built")
    response = client.get("/jobs/some-id/clips/other")
    assert response.status_code == 200
    assert "<div id=" in response.text or "<script" in response.text
