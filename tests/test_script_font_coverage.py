"""Captions must use a font that actually has the glyphs.

A caption set in a font with no glyph for the script renders as empty boxes, and
the clip is then useless in that language however well it was transcribed. The
fix has to be checked against fonts that are *installed*, because that is what
decides it.

The failure this pins: `SCRIPT_FONTS` named exactly one face per script, and the
obvious face is not always present. A machine without SimSun or MS Gothic sends
`resolve_font` to its last resort - any bold font in the directory - which is a
Latin face. Chinese, Japanese and Korean captions then render as boxes.

Checked against this machine's actual font directory rather than a list of what
Windows usually ships.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.render import renderer                  # noqa: E402

FONT_DIRS = [Path(p) for p in renderer.fg._FONT_DIRS]


def installed() -> set[str]:
    names: set[str] = set()
    for directory in FONT_DIRS:
        if directory.is_dir():
            names.update(p.name.lower() for p in directory.iterdir())
    return names


def resolve(name: str) -> Path | None:
    """The file `resolve_font` would actually land on, or None."""
    from pathlib import Path as P

    for directory in FONT_DIRS:
        if not directory.is_dir():
            continue
        for candidate in renderer.fg.FONT_CANDIDATES.get(name, [name]):
            path = P(directory) / candidate
            if path.is_file():
                return path
            if any(ch in candidate for ch in "*?[") and any(P(directory).glob(candidate)):
                return next(iter(sorted(P(directory).glob(candidate))))
    return None


def covers(name: str, codepoint: int) -> bool:
    """Does the resolved font file contain this character?

    Read straight out of the font's own character map - the same way a renderer
    will. Requires fontTools, so it skips rather than guessing.
    """
    path = resolve(name)
    if path is None:
        return False
    try:
        from fontTools.ttLib import TTCollection, TTFont
    except ImportError:
        pytest.skip("fontTools is not installed")
    if path.suffix.lower() == ".ttc":
        fonts = TTCollection(str(path), lazy=True).fonts
    else:
        fonts = [TTFont(str(path), fontNumber=0, lazy=True)]
    try:
        return any(codepoint in (f.getBestCmap() or {}) for f in fonts)
    finally:
        for f in fonts:
            try:
                f.close()
            except Exception:
                pass


# The languages by number of speakers, with one character from each script.
SCRIPTS = [
    ("latin", "Segoe UI Black", 0x0061),
    ("cyrillic", "Segoe UI", 0x0416),          # Russian
    ("arabic", "Segoe UI", 0x0645),            # Arabic
    ("cjk", None, 0x4E2D),                     # Chinese
    ("kana", None, 0x3042),                    # Japanese
    ("hangul", None, 0xAC00),                  # Korean
]


@pytest.mark.parametrize("script,expected,codepoint", SCRIPTS, ids=[s[0] for s in SCRIPTS])
def test_the_font_chosen_for_a_script_really_covers_it(script, expected, codepoint):
    sample = {
        "latin": "hello", "cyrillic": "\u0416\u0438\u0432\u043e\u0442",
        "arabic": "\u0645\u0631\u062d\u0628\u0627", "cjk": "\u4e2d\u6587\u5b57\u5e55",
        "kana": "\u3042\u3044\u3046\u3048\u304a", "hangul": "\uac00\ub138\ub2e4",
    }[script]
    from clipforge.engines import text as text_engine

    detected = text_engine.detect_scripts(sample)
    # Latin is the default and is deliberately not reported as a detected script,
    # so there is nothing to assert for it beyond the font covering the glyph.
    if script != "latin":
        assert script in detected, f"{script} not detected in {sample!r}"

    font = renderer.font_for_text(sample)
    if expected and font == expected:
        return                      # the known-good face is installed
    if not resolve(font):
        pytest.skip(f"no font on this machine covers {script}")
    assert covers(font, codepoint), (
        f"{script} picks {font}, which has no glyph for U+{codepoint:04X} - "
        f"the caption would render as empty boxes")


def test_every_script_lists_more_than_one_face():
    """A single named face is the bug; one is only ever enough by luck."""
    for script, faces in renderer.SCRIPT_FONTS.items():
        assert isinstance(faces, list), f"{script} still maps to a single name"
        assert len(faces) >= 2, f"{script} has only one candidate: {faces}"


def test_cjk_prefers_a_cjk_face_over_the_latin_default():
    font = renderer.font_for_text("\u4e2d\u6587\u5b57\u5e55")
    assert font != "Segoe UI Black", "Latin display face chosen for Chinese"
    if resolve(font):
        assert covers(font, 0x4E2D)


@pytest.mark.parametrize("sample,script", [
    ("\u0645\u0631\u062d\u0628\u0627", "arabic"),
    ("\u0416\u0438\u0432\u043e\u0442", "cyrillic"),
    ("\u4e2d\u6587", "cjk"),
    ("\u3042\u3044\u3046", "kana"),
    ("\uac00\ub2e4", "hangul"),
    ("\u0915\u0932", "devanagari"),
    ("\u0e01\u0e32", "thai"),
])
def test_detection_and_selection_agree(sample, script):
    """A script that is detected must pull the selection away from Latin."""
    font = renderer.font_for_text(sample)
    faces = renderer.SCRIPT_FONTS.get(script, [])
    installed_faces = [f for f in faces if renderer._font_exists(f)]
    if not installed_faces:
        pytest.skip(f"this machine has no font covering {script}")
    assert font in installed_faces, f"{script}: chose {font}, have {installed_faces}"


def test_the_first_installed_face_wins_even_when_earlier_ones_are_missing(monkeypatch):
    """The existence check itself.

    On this machine every *first* candidate happens to be installed, so nothing
    else here distinguishes "first installed" from "first named". Removing the
    check entirely passes all the other tests - which is why this one patches in
    a face that is definitely not present and asserts the next one is used.
    """
    monkeypatch.setitem(renderer.SCRIPT_FONTS, "cjk",
                        ["Definitely Not Installed", "Yu Gothic UI", "Segoe UI"])
    chosen = renderer.font_for_text("\u4e2d\u6587")
    assert chosen == "Yu Gothic UI", (
        f"chose {chosen!r}; a missing face must be skipped, not returned")
    assert resolve(chosen) is not None


def test_a_script_with_nothing_installable_keeps_the_default(monkeypatch):
    """No covering face at all must not produce a font that does not exist."""
    monkeypatch.setitem(renderer.SCRIPT_FONTS, "cjk", ["Nope One", "Nope Two"])
    assert renderer.font_for_text("\u4e2d\u6587") == "Segoe UI Black"


def test_latin_text_is_unaffected():
    """The common case must not change."""
    assert renderer.font_for_text("hello there") == "Segoe UI Black"
    assert renderer.font_for_text("") == "Segoe UI Black"


def test_mixed_script_picks_one_that_exists():
    font = renderer.font_for_text("hello \u0645\u0631\u062d\u0628\u0627 world")
    assert resolve(font) is not None
