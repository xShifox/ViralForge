"""The strip has to appear, disappear, and never touch anything else.

These build the graph and read it back. They do not run ffmpeg: what is being
checked is that the fragments are well-formed, that the labels chain correctly,
and - the important one - that a clip with no inserts produces exactly the graph
it produced before this feature existed.

That last property is what protects the 4 GB card. If adding no visuals produced
even a slightly different filtergraph, every existing render would change, and
the guarantee that two cards produce the same file would be untestable.
"""
from __future__ import annotations

from pathlib import Path

from clipforge.render import brollgraph


def _insert(**kwargs) -> brollgraph.Insert:
    base = {"path": Path("C:/lib/goal.mp4"), "input_index": 1,
            "at": 0.0, "duration": 3.0, "kind": "video"}
    base.update(kwargs)
    return brollgraph.Insert(**base)


# ------------------------------------------------------------------ geometry

def test_the_strip_sits_across_the_bottom_of_the_frame():
    box = brollgraph.strip_geometry(1080, 1920)
    assert box["width"] == 1080
    assert box["y"] + box["height"] == 1920, "the strip must reach the bottom edge"
    assert box["y"] > 0, "it must not cover the whole frame"


def test_the_strip_leaves_the_speaker_room():
    box = brollgraph.strip_geometry(1080, 1920)
    assert box["height"] <= 1920 * 0.45, box


# ------------------------------------------------------------------- nothing

def test_no_inserts_leaves_the_graph_exactly_as_it_was():
    """The property that makes this safe to ship."""
    parts, label = brollgraph.insert_filters([], 1080, 1920, current="sub")
    assert parts == []
    assert label == "sub", "the chain label must not change when nothing is inserted"


def test_no_inserts_means_no_extra_inputs():
    assert brollgraph.input_arguments([]) == []


def test_no_inserts_produces_no_summary():
    assert brollgraph.describe([], 1080, 1920) == ""


# -------------------------------------------------------------------- videos

def test_a_shot_becomes_a_trimmed_input():
    parts, label = brollgraph.insert_filters([_insert()], 1080, 1920)
    joined = ";".join(parts)
    assert "trim=start=0.000:duration=3.000" in joined
    assert label != "base", "the overlay must hand on a new label"


def test_a_shot_is_only_visible_during_its_own_moment():
    parts, _ = brollgraph.insert_filters([_insert(at=4.0, duration=3.0)],
                                         1080, 1920)
    joined = ";".join(parts)
    assert "between(t,4.000,7.000)" in joined, joined[-300:]


def test_a_shot_is_sized_to_the_strip():
    parts, _ = brollgraph.insert_filters([_insert()], 1080, 1920)
    box = brollgraph.strip_geometry(1080, 1920)
    assert f"crop={box['width']}:{box['height']}" in ";".join(parts)


def test_a_shot_fades_in_and_out():
    joined = ";".join(brollgraph.insert_filters([_insert(duration=3.0)], 1080, 1920)[0])
    assert "fade=t=in" in joined
    assert "fade=t=out" in joined


def test_a_short_shot_does_not_fade_out_before_it_has_appeared():
    """A fade out starting before the shot does would show it never arriving."""
    parts, _ = brollgraph.insert_filters([_insert(duration=0.25)], 1080, 1920)
    joined = ";".join(parts)
    assert "fade=t=out:st=0.050:d=0.2" in joined, joined[-300:]


# --------------------------------------------------------------------- stills

def test_a_still_is_looped_rather_than_trimmed():
    """A one-frame image has no duration to trim, so it is looped instead."""
    parts, _ = brollgraph.insert_filters([_insert(kind="image")], 1080, 1920)
    joined = ";".join(parts)
    assert "loop=loop=-1" in joined
    assert "trim=start=" not in joined.split("[raw")[0], \
        "a still must not be trimmed like a video"


def test_a_still_and_a_clip_get_the_same_scaling():
    """Otherwise the library looks different depending on file type."""
    video = ";".join(brollgraph.insert_filters([_insert(kind="video")], 1080, 1920)[0])
    still = ";".join(brollgraph.insert_filters([_insert(kind="image")], 1080, 1920)[0])
    box = brollgraph.strip_geometry(1080, 1920)
    fragment = f"scale={box['width']}:{box['height']}"
    assert fragment in video and fragment in still


# ------------------------------------------------------------------ several

def test_several_shots_chain_and_each_hands_on_a_label():
    shots = [_insert(input_index=1, at=0.0), _insert(input_index=2, at=4.0)]
    parts, label = brollgraph.insert_filters(shots, 1080, 1920)
    joined = ";".join(parts)
    assert "between(t,0.000,3.000)" in joined
    assert "between(t,4.000,7.000)" in joined
    assert label == "ov1", label


def test_the_inputs_are_numbered_in_the_order_given():
    shots = [_insert(input_index=3, path=Path("C:/lib/a.mp4")),
             _insert(input_index=1, path=Path("C:/lib/b.mp4"))]
    args = brollgraph.input_arguments(shots)
    # sorted by input index, so the caller can append them in that order
    assert args[1].endswith("b.mp4"), args
    assert args[3].endswith("a.mp4"), args


def test_the_summary_says_what_was_shown_and_how_big():
    text = brollgraph.describe([_insert(path=Path("C:/lib/goal.mp4"))], 1080, 1920)
    box = brollgraph.strip_geometry(1080, 1920)
    assert "1 related visual" in text
    assert f"{box['height']}px" in text


def test_two_files_are_counted_separately():
    shots = [_insert(path=Path("C:/lib/a.mp4")), _insert(path=Path("C:/lib/b.mp4"))]
    assert "from 2 files" in brollgraph.describe(shots, 1080, 1920)
