"""Per-clip undo history: a JSONL stack of pre-edit snapshots."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from ..core.models import Clip

MAX_DEPTH = 20

_locks: dict[str, threading.Lock] = {}
_lock_of_locks = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    import hashlib
    key = hashlib.sha256(str(path).encode()).hexdigest()
    with _lock_of_locks:
        return _locks.setdefault(key, threading.Lock())


def _file(root, clip_id: str) -> Path:
    return Path(root) / f"{clip_id}.jsonl"


def _read(root, clip_id: str) -> list[dict]:
    try:
        lines = _file(root, clip_id).read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _write(root, clip_id: str, stack: list[dict]) -> None:
    path = _file(root, clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in stack) + ("\n" if stack else ""),
                    encoding="utf-8")
    os.replace(tmp, path)


def depth(root, clip_id: str) -> int:
    return len(_read(root, clip_id))


def push(root, clip: Clip) -> None:
    # Lock around read-append-write: without it two threads push/pop
    # interleave and one undo step is silently lost.
    with _lock_for(_file(root, clip.id)):
        stack = _read(root, clip.id)
        stack.append(clip.model_dump(mode="json"))
        while len(stack) > MAX_DEPTH:
            stack.pop(0)
        _write(root, clip.id, stack)


def pop(root, clip_id: str):
    with _lock_for(_file(root, clip_id)):
        stack = _read(root, clip_id)
        if not stack:
            return None
        payload = stack.pop()
        _write(root, clip_id, stack)
        return payload


def peek(root, clip_id: str):
    stack = _read(root, clip_id)
    return stack[-1] if stack else None


def clear(root, clip_id: str) -> None:
    with _lock_for(_file(root, clip_id)):
        try:
            _file(root, clip_id).unlink(missing_ok=True)
        except OSError:
            pass


def restore(clip: Clip, payload) -> None:
    """Restore a snapshot onto clip, marking it stale (needs a re-render)."""
    if payload is None:
        return
    from ..core.models import ClipState

    data = dict(payload)
    clip_id = clip.id
    data["id"] = clip_id
    restored = Clip.model_validate(data)
    for name in Clip.model_fields:
        if name == "id":
            continue
        setattr(clip, name, getattr(restored, name))
    clip.state = ClipState.stale
    clip.id = clip_id
