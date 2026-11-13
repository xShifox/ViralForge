"""Simple authentication token management for ViralForge."""

from __future__ import annotations

import os
import secrets
import threading
from pathlib import Path


_lock = threading.Lock()


def token_path(data_dir: Path) -> Path:
    """Return the path to the auth token file."""
    return data_dir / "auth_token"


def load_or_create(data_dir: Path, regenerate: bool = False) -> str:
    """Load the existing token or create a new one.

    Atomic + locked: two threads creating at once previously each got a
    different token than the file held. The write goes via a temp file
    and ``os.replace`` with 0o600 permissions.
    """
    path = token_path(data_dir)
    with _lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        if regenerate or not path.is_file():
            token = secrets.token_urlsafe(32)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(token, encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, path)
            return token
        return path.read_text(encoding="utf-8").strip()


def invalidate(data_dir: Path | None = None) -> bool:
    """Invalidate the current token by removing it.

    Returns True when a token was removed. ``data_dir`` is accepted for
    callers that hold it; when omitted the token cannot be located and
    False is returned (the old stub always succeeded silently, leaving
    the token valid after "revocation").
    """
    if data_dir is None:
        return False
    path = token_path(data_dir)
    with _lock:
        try:
            existed = path.exists()
            path.unlink(missing_ok=True)
            return existed
        except OSError:
            return False


def matches(request_token: str, valid_token: str) -> bool:
    """Check if the request token matches the valid token."""
    if not isinstance(request_token, str) or not isinstance(valid_token, str):
        return False
    if not request_token or not valid_token:
        return False
    return secrets.compare_digest(request_token, valid_token)


def current(data_dir: Path) -> str:
    """Get the current token, creating one if it doesn't exist."""
    return load_or_create(data_dir)