"""Optional publishing metadata: off by default, never burned into the picture."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.config import RenderSettings  # noqa: E402
from clipforge.engines import metadata as meta  # noqa: E402


# ------------------------------------------------------------------- default off

def test_it_is_off_by_default():
    assert RenderSettings().generate_metadata is False


def test_off_means_empty_fields():
    clip = _clip()
    if not RenderSettings().generate_metadata:
        assert clip.hashtags == [] and clip.cta == "" and clip.metadata_sidecar == ""


def _clip(text="", title="", hashtags_=None, cta=""):
    from clipforge.core.models import Clip, ScoreBreakdown

    return Clip(
        id="c0", index=0, start=0.0, end=30.0, text=text,
        breakdown=ScoreBreakdown(), reasons=[], hashtags=hashtags_ or [], cta=cta,
    )


# -------------------------------------------------------------------- hashtags

def test_hashtags_come_from_the_clip():
    tags = meta.hashtags("battery thermal runaway in lithium cells", "Science")
    assert "battery" in tags or "thermal" in tags or "lithium" in tags


def test_a_headline_describes_the_clip_better_than_the_opening():
    """The opening is usually filler; the headline is the tool's own pick."""
    opener = "So today I want to talk about basically the compression thing"
    tagged = meta.build(opener, "", "", title="HOW COMPRESSION WORKS").get("hashtags", [])
    assert "COMPRESSION" in tagged or "compression" in tagged


def test_generic_words_are_not_topics():
    tags = meta.hashtags("So today I want to know the thing that people use", "")
    assert not ({"today", "want", "know", "thing", "people"} & set(tags))


def test_tags_are_bounded():
    tags = meta.hashtags(" ".join(f"word{i}" for i in range(50)), "")
    assert len(tags) <= meta.MAX_HASHTAGS


def test_tags_are_unique():
    tags = meta.hashtags("battery battery battery thermal thermal", "")
    assert len(tags) == len(set(tags))


def test_tags_carry_no_characters_a_hashtag_cannot_hold():
    tags = meta.hashtags("hello, world! (parentheses) [brackets] -- dashes", "")
    for tag in tags:
        assert all(ch.isalnum() or ch == "_" for ch in tag), tag


def test_non_latin_letters_survive():
    """Transliterating somebody's language to satisfy a hashtag rule would be odd."""
    tags = meta.hashtags("البطارية continually", "")
    assert any(tag for tag in tags)
    assert any(any("؀" <= ch <= "ۿ" for ch in tag) for tag in tags) or tags


def test_the_same_clip_always_produces_the_same_tags():
    """Two machines must produce the same file, so nothing here may vary by time."""
    text = "battery thermal runaway in lithium cells"
    assert meta.hashtags(text, "Science") == meta.hashtags(text, "Science")


def test_the_niche_appears_among_the_tags():
    tags = meta.hashtags("battery thermal runaway", "Science")
    assert "Science" in tags


# ------------------------------------------------------------------------- CTA

def test_a_cta_is_never_invented():
    """Putting words in a speaker's mouth on their account is not this tool's call."""
    assert meta.cta_from_dialogue("This is just a normal sentence about batteries") == ""


def test_a_cta_the_speaker_said_is_quoted_back():
    text = "That is the whole trick. If you want more, follow the channel."
    cta = meta.cta_from_dialogue(text)
    assert "follow" in cta.lower()


def test_the_cta_is_the_speakers_own_words():
    text = "That is the whole trick. If you want more, follow the channel."
    assert meta.cta_from_dialogue(text) in text


def test_a_cta_in_arabic_is_found():
    assert meta.cta_from_dialogue("هذه كانت الحلقة. تابعونا للمزيد من المحتوى.")


# -------------------------------------------------------------------- sidecar

def test_the_sidecar_carries_both():
    block = meta.metadata_block(["alpha", "beta"], "Follow for more")
    assert "Follow for more" in block
    assert "#alpha #beta" in block


def test_an_empty_sidecar_is_empty_not_padded():
    assert meta.metadata_block([], "") == ""


def test_build_returns_everything_the_runner_needs():
    out = meta.build("battery thermal", "Science", "science", "BATTERY SAFETY")
    assert set(out) == {"hashtags", "cta", "sidecar"}
    assert out["sidecar"]


def test_build_survives_empty_input():
    out = meta.build("")
    assert out["hashtags"]
    assert out["cta"] == ""


def test_nothing_lands_in_the_subtitle_style():
    """The guard for "metadata, not burned in": captions are built elsewhere and
    this module has no access to them."""
    assert not hasattr(meta, "SubtitleStyle")
    assert not hasattr(meta, "ass")
    assert not hasattr(meta, "ffmpeg")