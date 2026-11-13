"""The interface must not be served stale, and the bundles must not be re-fetched.

Two halves of one thing. ``index.html`` names the hashed bundles, so a cached copy
of it points at a build that may no longer exist - and a cached copy is what the
browser serves when the response carries no Cache-Control at all, which is the
default here. The failure is silent: the server is serving the new interface, the
screen shows the old one, and there is nothing to see in a log.

The other half is the reason not to just set ``no-store`` everywhere. Asset
filenames carry a content hash, so ``index-Ab12Cd.js`` can only ever be one build's
content. Caching those for a year is safe, and without it every launch
re-downloads the whole bundle.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WEB_DIST = ROOT / "web" / "dist"

pytestmark = pytest.mark.skipif(
    not (WEB_DIST / "index.html").is_file(),
    reason="the interface is not built",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from clipforge.app import create_app

    return TestClient(create_app())


def test_index_html_must_be_revalidated(client):
    """Otherwise a browser keeps pointing at a bundle from a build that is gone."""
    headers = client.get("/").headers
    assert "no-cache" in headers.get("cache-control", ""), (
        "index.html is served with no Cache-Control, so the browser guesses "
        f"freshness from Last-Modified and may keep the old one: {dict(headers)}"
    )


def test_a_client_side_route_also_gets_no_cache(client):
    """Any path that falls through to index.html needs the same treatment."""
    headers = client.get("/some/deep/route").headers
    assert "no-cache" in headers.get("cache-control", "")


def test_hashed_assets_are_cached_hard(client):
    """Safe precisely because the filename changes when the content does."""
    assets = sorted((WEB_DIST / "assets").glob("*.js"))
    assert assets, "no built assets to check"
    response = client.get(f"/assets/{assets[0].name}")
    assert response.status_code == 200
    cache = response.headers.get("cache-control", "")
    assert "max-age=" in cache and "immutable" in cache, cache


def test_the_vram_bar_is_one_bar_not_four():
    """Four adjacent fills with no gaps read as four bars.

    Checked against the built bundle, because that is what the browser actually
    runs - a change to the component that never reaches a build is not a fix.
    """
    bundle = max((WEB_DIST / "assets").glob("*.js"), key=lambda p: p.stat().st_mtime)
    text = bundle.read_text(encoding="utf-8", errors="ignore")
    assert "vram-fill" in text, "the single-bar class is not in the built bundle"
    for gone in ("vram-ours", "vram-other", "vram-reserved"):
        assert gone not in text, f"{gone} is still in the bundle - the bar is still split"


def test_the_build_has_no_leftover_segment_styles():
    css = max((WEB_DIST / "assets").glob("*.css"), key=lambda p: p.stat().st_mtime)
    text = css.read_text(encoding="utf-8", errors="ignore")
    assert ".vram-fill" in text
    assert ".vram-seg" not in text, "the old per-segment styles are still in the build"