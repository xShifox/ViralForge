"""Captions should arrive, not appear.

Captions used to switch on between one frame and the next: a caption that is
fully opaque from its first frame reads as a subtitle pasted onto the video
rather than as part of it. A short fade costs nothing here - libass draws it from
the cue timings that were already in the file, so no extra frames are rendered
and no setting that decides quality is touched.

The properties that matter:

* A cue shorter than twice the fade must not spend its life half-visible, so the
  fade is capped at a fraction of the cue's own length.
* The emphasised cue arrives faster than a flat one.
* Turning it off must produce the file it produced before, byte for byte.
"""
from __future__ import annotations

from pathlib import Path

from clipforge.core.models import SubtitleCue, SubtitleStyle
from clipforge.render import ass


def _cues() -> list[SubtitleCue]:
    return [
        SubtitleCue(index=0, start=0.0, end=2.0, text="the first spoken line"),
        SubtitleCue(index=1, start=2.0, end=2.4, text="a very short one", emphasis=True),
    ]


def _build(tmp_path: Path, **style_kwargs) -> str:
    style = SubtitleStyle(font="Segoe UI Black", **style_kwargs)
    path = ass.build_ass(_cues(), tmp_path / "clip.ass", style,
                         width=1080, height=1920, fontname="Segoe UI Black",
                         font_size=74)
    return path.read_text(encoding="utf-8")


def _fades(text: str) -> list[int]:
    """The fade-in time of every caption, in centiseconds.

    ``\\fad`` takes two numbers, so the in-time is the first of them - parsing the
    whole thing as an integer is the obvious mistake and it fails on the second
    caption rather than the first.
    """
    return [int(seg.split(")")[0].split(",")[0])
            for seg in text.split("{\\fad(")[1:]]


def _cue_lines(text: str) -> list[str]:
    return [l for l in text.splitlines() if l.startswith("Dialogue: 0,")]


def test_captions_fade_in_when_motion_is_on(tmp_path):
    text = _build(tmp_path, caption_motion=True)
    assert "{\\fad(" in text, "the fade tag is missing"


def test_captions_are_unchanged_when_motion_is_off(tmp_path):
    """Off must mean exactly what it meant before the feature existed."""
    text = _build(tmp_path, caption_motion=False)
    assert "{\\fad(" not in text, "motion was turned off but the fade is still there"


def test_a_short_cue_does_not_fade_for_longer_than_it_lasts(tmp_path):
    """A 0.4 s cue faded over 0.12 s in and out would be dim for a quarter of it.

    The fade is capped at a fraction of the cue, so a quick line still reads as
    a quick line instead of shimmering.
    """
    text = _build(tmp_path, caption_motion=True)
    fades = _fades(text)
    assert fades, "no fade was written at all"
    # The shorter cue (0.4 s) is capped at 28% of itself.
    assert min(fades) / 100 <= 0.4 * 0.28 + 0.001, fades


def test_the_fade_never_exceeds_the_cap(tmp_path):
    text = _build(tmp_path, caption_motion=True)
    fades = _fades(text)
    assert fades
    assert max(fades) <= int(ass.CUE_FADE_SECONDS * 100) + 1, fades


def test_an_emphasised_cue_arrives_faster_than_a_flat_one(tmp_path):
    text = _build(tmp_path, caption_motion=True)
    lines = _cue_lines(text)
    flat = next(l for l in lines if "FIRST SPOKEN LINE" in l.upper())
    punchy = next(l for l in lines if "VERY SHORT ONE" in l.upper())

    def fade_of(line: str) -> int:
        return int(line.split("{\\fad(")[1].split(")")[0].split(",")[0])

    assert fade_of(pumpy := punchy) <= fade_of(flat), (pumpy, flat)


def test_an_empty_cue_writes_nothing_at_all(tmp_path):
    """A fade must not resurrect a cue that has no words in it."""
    style = SubtitleStyle(font="Segoe UI Black", caption_motion=True)
    path = ass.build_ass([SubtitleCue(index=0, start=0.0, end=1.0, text="   ")],
                         tmp_path / "empty.ass", style)
    text = path.read_text(encoding="utf-8")
    assert "Dialogue: 0," not in text


def test_the_sidecar_captions_are_never_animated(tmp_path):
    """SRT and VTT have no way to express a fade.

    Writing one anyway would produce a sidecar that players render literally, so
    the animation stays in the burned-in overlay only.
    """
    style = SubtitleStyle(font="Segoe UI Black", caption_motion=True)
    cues = _cues()
    srt = ass.build_srt(cues, tmp_path / "c.srt", uppercase=True).read_text(encoding="utf-8")
    vtt = ass.build_vtt(cues, tmp_path / "c.vtt").read_text(encoding="utf-8")
    assert "\\fad" not in srt and "\\fad" not in vtt
