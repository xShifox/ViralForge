"""Test-wide configuration.

The `slow` marker is registered here rather than in pytest.ini because pytest 9
silently ignores a `markers` key in that file, which left the launcher tests
emitting an "unknown mark" warning on every run. conftest is honoured by every
pytest version.

Run the fast suite with:  pytest -m "not slow"
"""
from __future__ import annotations

import os

import pytest

AUTH_DISABLE_ENV = "CLIPFORGE_DISABLE_AUTH"


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: starts a real server or child process; deselect with -m 'not slow'",
    )
    config.addinivalue_line(
        "markers",
        "authtoken: runs with API authentication enforced; the rest of the "
        "suite runs with it disabled",
    )
    # The default has to be established here, before collection: several test
    # modules build their TestClient at *import* time, which is earlier than any
    # fixture can run, and `create_app` reads this at construction time.
    os.environ.setdefault(AUTH_DISABLE_ENV, "1")


@pytest.fixture(autouse=True)
def _api_auth_bypass(request) -> None:
    """Turn the API token off for every test except the ones about the token.

    1,200 unrelated tests each having to fetch a token first would prove nothing
    and cost a great deal, so they run unauthenticated. `tests/test_api_auth.py`
    marks itself `authtoken` and therefore gets the real thing.

    This is a fixture rather than an environment variable cleared once during
    collection, because it has to be per-test. Doing it at collection time
    cleared the bypass for the *whole session* the moment a single marked module
    was collected - so every other API test silently ran against an
    authenticated app, and only the one test that happened to assert on a
    status code noticed.

    The unmarked branch sets rather than `setdefault`s, on purpose: the pop above
    is a raw `os.environ.pop`, not a monkeypatch, so it is *not* undone when the
    marked test finishes. Without an unconditional set on the way back, a test
    that ran after `test_api_auth.py` would inherit an authenticated app -
    which is exactly what happened, and it made the suite order-dependent.
    """
    if request.node.get_closest_marker("authtoken"):
        os.environ.pop(AUTH_DISABLE_ENV, None)
    else:
        os.environ[AUTH_DISABLE_ENV] = "1"
