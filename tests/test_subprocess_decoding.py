"""Pin the decoding on every subprocess call that reads a tool's output.

These call sites all sit inside a blanket ``except``, so a decode failure does
not crash the application - it degrades it. That is the worse outcome: a card
whose name the locale codec cannot read is reported as no card at all, and the
job quietly renders in software with nothing in the log but a debug line. Every
other ffmpeg call in this project was pinned to utf-8 after the Arabic work;
these were missed because their output looked machine-generated.

The tests are behavioural rather than a grep. ``subprocess.run`` is patched on
the shared module, because one of these imports it inside the function, and the
stand-in refuses to hand back text unless the caller said how to decode - which
is exactly the condition under which the real one raises deep in a reader thread.
"""
from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge import gpu                                    # noqa: E402
from clipforge.engines import asr                            # noqa: E402
from clipforge.render import renderer                        # noqa: E402


class _RefusesWithoutDecoding:
    """A stand-in for ``subprocess.run`` that enforces the rule."""

    def __init__(self, payload: bytes, returncode: int = 0) -> None:
        self.payload = payload
        self.returncode = returncode
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("capture_output") and "encoding" not in kwargs:
            raise UnicodeDecodeError("charmap", self.payload, 0, 1,
                                     "character maps to <undefined>")
        return subprocess.CompletedProcess(
            args=cmd, returncode=self.returncode,
            stdout=self.payload.decode("utf-8", "replace"), stderr="")


@pytest.fixture
def patch_run(monkeypatch):
    def install(payload: bytes) -> _RefusesWithoutDecoding:
        fake = _RefusesWithoutDecoding(payload)
        monkeypatch.setattr(subprocess, "run", fake)
        return fake
    return install


def test_a_gpu_name_the_locale_cannot_read_is_still_parsed(patch_run):
    """Otherwise the card is reported missing and the job renders in software.

    The name ends in U+0081, a byte cp1252 leaves undefined - what a localised
    or vendor-specific device name amounts to. It goes on the *name* field, not
    the number: ``--format=csv,noheader,nounits`` means the memory figure is a
    bare integer, and putting the stray byte there would fail the parse for an
    unrelated reason and prove nothing about decoding.
    """
    payload = "NVIDIA GeForce RTX 4060 Laptop GPU \u0081, 4096".encode("utf-8")
    patch_run(payload)

    name, total, _has = gpu._query()

    assert "RTX 4060" in name, (
        f"the card was reported as absent because its name was unreadable: {name!r}")
    assert total == 4096, f"the memory reading was lost with it: {total}"


def test_the_encoder_list_is_read_even_when_it_is_not_plain_ascii(patch_run):
    payload = "Encoders:\n V..... h264_nvenc  \u00a0NVIDIA NVENC H264\n".encode("utf-8")
    patch_run(payload)

    _name, _total, has_nvenc = gpu._query()

    assert has_nvenc, "h264_nvenc was missed, so GPU encoding would be switched off"


def test_the_memory_reading_survives_a_messy_one(patch_run):
    # Every field here is an integer, so there is nowhere to hide a stray byte on
    # the line that gets parsed - and the parser is all-or-nothing: one bad field
    # and it returns four zeros. So the byte rides on the *second* line, the shape
    # a machine with two cards reports. The whole buffer still has to decode
    # before any of it can be read, which is the thing under test.
    payload = ("4096, 512, 3584, 7\n"
               "4096, 512, 3584, 7 \u0081\n").encode("utf-8")
    patch_run(payload)

    total, _used, free, _util = gpu._memory_query()

    assert total == 4096 and free == 3584, (
        f"the card's memory was misread as {total}/{free} rather than 4096/3584")


def test_free_vram_survives_an_undecodable_reading(patch_run):
    # ``-i 0`` names one card, so there is normally one line. The second line
    # here stands in for a machine where the driver prints more than was asked
    # for - it still has to decode before the first line can be read.
    patch_run(("4096\n512 \u0081\n").encode("utf-8"))

    # Not "in (4096, None)". That passed before the fix as well, because the
    # blanket except turns a decode failure into None - which is the whole fault.
    # An honest "could not read" is acceptable here; a value the caller cannot
    # tell from a real reading is not.
    assert asr._free_vram_mb() == 4096, (
        "free VRAM came back as 'unknown', so the encoder is told the card is "
        "full on a card with 4 GB free")


def test_the_renderer_agrees_the_card_can_encode_after_a_messy_reading(patch_run, monkeypatch):
    patch_run(" V..... h264_nvenc \u0081\n".encode("utf-8"))
    monkeypatch.setattr(renderer, "_nvenc_checked", None)

    assert renderer.nvenc_available() is True, (
        "an unreadable encoder list silently downgraded the job to software")


def test_every_reading_subprocess_call_asks_how_to_decode():
    """The rule, stated once, over the files that read a tool's output.

    A behavioural test proves one call site at a time. This proves the pattern
    holds, so the next person who adds a query does not reintroduce the gap -
    which is exactly what happened with the calls fixed here.
    """
    offenders = []
    for name in ("clipforge/gpu.py", "clipforge/engines/asr.py",
                 "clipforge/render/renderer.py", "clipforge/render/ffmpeg.py",
                 "clipforge/core/compliance.py"):
        path = ROOT / name
        if not path.is_file():
            continue
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute)
                    and fn.attr in ("run", "Popen", "check_output")):
                continue
            kw = {k.arg for k in node.keywords if k.arg}
            if "text=True" in ast.unparse(node) and "encoding" not in kw:
                offenders.append(f"{name}:{node.lineno}")
    assert not offenders, (
        "subprocess calls that read a tool's output without saying how to decode "
        "it, so they fall back to the locale codec and lose it on any machine "
        "whose locale cannot read the output:\n  " + "\n  ".join(offenders))
