"""B-roll strip: insert related visuals as a bottom strip of the frame.

The strip sits across the bottom, fades in and out, and hands on a new label
each time. With no inserts the graph is byte-for-byte what it was before the
feature existed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Insert:
    path: Path = None
    input_index: int = 0
    at: float = 0.0
    duration: float = 3.0
    kind: str = "video"


def strip_geometry(width: int, height: int) -> dict:
    """Strip rectangle: full width, bottom-anchored, at most 45% of height."""
    strip_height = max(1, int(height * 0.3))
    strip_height = min(strip_height, int(height * 0.45))
    return {"width": width, "height": strip_height, "y": height - strip_height}


def insert_filters(inserts: list[Insert], width: int, height: int,
                   current: str = "base") -> tuple[list[str], str]:
    """Build overlay fragments chaining from ``current`` to a fresh label."""
    if not inserts:
        return [], current
    box = strip_geometry(width, height)
    parts: list[str] = []
    label = current
    for number, insert in enumerate(sorted(inserts, key=lambda i: i.input_index), start=1):
        raw = f"raw{number}"
        out = f"ov{number}"
        at = float(insert.at)
        dur = max(0.1, float(insert.duration))
        if insert.kind == "image":
            parts.append(
                f"[{insert.input_index}:v]loop=loop=-1:size=1,"
                f"scale={box['width']}:{box['height']}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={box['width']}:{box['height']}:(ow-iw)/2:(oh-ih)/2:black,"
                f"trim=start={at:.3f}:duration={dur:.3f}[{raw}]"
            )
        else:
            parts.append(
                f"[{insert.input_index}:v]trim=start={at:.3f}:duration={dur:.3f},"
                f"scale={box['width']}:{box['height']}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={box['width']}:{box['height']}:(ow-iw)/2:(oh-ih)/2:black[{raw}]"
            )
        fade_in_end = min(0.2, dur / 2.0)
        fade_out_start = max(fade_in_end, dur - 0.2)
        parts.append(
            f"[{label}][{raw}]overlay=x=0:y={box['y']}:format=auto:"
            f"enable='between(t,{at:.3f},{at + dur:.3f})',"
            f"fade=t=in:st={at:.3f}:d={fade_in_end:.3f},"
            f"fade=t=out:st={at + fade_out_start:.3f}:d=0.2[{out}]"
        )
        # The overlay's enable chooses visibility; fades are part of the chain.
        label = out
    return parts, label


def input_arguments(inserts: list[Insert]) -> dict[int, str]:
    """Map input_index -> path, in the order the caller appends them."""
    out: dict[int, str] = {}
    for insert in sorted(inserts or [], key=lambda i: i.input_index):
        out[int(insert.input_index)] = str(insert.path)
    return out


def describe(inserts: list[Insert], width: int, height: int) -> str:
    """One-line human summary."""
    if not inserts:
        return ""
    box = strip_geometry(width, height)
    count = len(inserts)
    files = len({str(i.path) for i in inserts})
    noun = "related visual" if count == 1 else "related visuals"
    source = "" if files == count else (f" from {files} files" if files > 1 else "")
    return f"{count} {noun}{source} ({box['height']}px strip)"
