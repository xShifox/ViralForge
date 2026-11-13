"""Settings have to survive being saved and read back.

The failure this covers is invisible from inside a running process: `save_settings`
hands back the in-memory object it was given, so a setting the user just changed
appears to work, and only comes back wrong the next time the server starts. Every
option inside `asr`, `detect`, `render` and `brand` lived in that blind spot,
because the overlay wrote into a throwaway copy of the settings rather than onto
the settings themselves.

`asdict()` is recursive, which is the whole reason: a nested dataclass comes back
from it as a plain `dict`, so the branch that recursed into a dataclass was dead,
and the dict branch mutated the copy. Top-level scalars were unaffected, which is
why it looked like a partial bug rather than a total one.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from dataclasses import fields
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge import config                      # noqa: E402


def test_every_setting_survives_a_save_and_reload():
    """The round trip, on a value touched at each level."""
    written = config.AppSettings()
    written.render.track_talker = True
    written.render.layout = "fit"
    written.render.video_bitrate = "12M"
    written.asr.device = "cpu"
    written.asr.require_gpu = False
    written.asr.beam_size = 7
    written.detect.max_clips = 11
    written.detect.score_threshold = 0.51
    written.brand.logo_opacity = 0.25
    written.brand.watermark_anchor = "top-right"
    written.output_folder = "C:/somewhere"
    written.max_upload_mb = 4321
    written.keep_intermediate = False
    written.platforms = ["shorts"]

    payload = json.loads(json.dumps(dataclasses.asdict(written)))
    loaded = config._apply(config.AppSettings(), payload)

    assert loaded == written, (
        "settings came back different:\n"
        + "\n".join(_differences(written, loaded)))


def _differences(want, got, prefix="") -> list[str]:
    out = []
    for spec in fields(want):
        a, b = getattr(want, spec.name), getattr(got, spec.name)
        if dataclasses.is_dataclass(a):
            out.extend(_differences(a, b, f"{prefix}{spec.name}."))
        elif a != b or type(a) is not type(b):
            out.append(f"{prefix}{spec.name}: saved {a!r} ({type(a).__name__}), "
                       f"loaded {b!r} ({type(b).__name__})")
    return out


@pytest.mark.parametrize("section", ["asr", "detect", "render", "brand"])
def test_each_nested_section_is_really_applied(section):
    """One per section, so a future break names the section it broke.

    The changed value has to suit the field's type: writing the string
    "changed" into an int field is correctly rejected by coercion, and a test
    that did that would be asserting that coercion does not work.
    """
    target = config.AppSettings()
    nested = getattr(target, section)
    original = getattr(nested, fields(nested)[0].name)
    changed = (not original if isinstance(original, bool)
               else original + 1 if isinstance(original, (int, float))
               else str(original) + "_changed")

    payload = dataclasses.asdict(target)
    payload[section][fields(nested)[0].name] = changed
    loaded = config._apply(config.AppSettings(), payload)
    assert getattr(getattr(loaded, section), fields(nested)[0].name) == changed, section


def test_the_overlay_is_not_writing_into_asdict_copy():
    """The specific mechanism, so the diagnosis is pinned and not just the symptom."""
    root = config.AppSettings()
    # asdict is recursive: this is why the dataclass branch was unreachable.
    dumped = dataclasses.asdict(root)
    assert isinstance(dumped["render"], dict)
    assert not hasattr(dumped["render"], "__dataclass_fields__")

    before = root.render.layout
    payload = {"render": {"layout": "fit"}}
    config._apply(root, payload)
    assert root.render.layout == "fit"
    # and the live object must still be a dataclass, not replaced by a dict
    assert dataclasses.is_dataclass(root.render)
    assert before != "fit"


def test_string_annotations_are_coerced_to_real_types():
    """`from __future__ import annotations` makes field types strings.

    Comparing ``'bool'`` against ``(int, float, bool, str)`` is always false, so
    the coercion never ran and a hand-edited settings file kept whatever type the
    JSON happened to hold. ``"keep_intermediate": "false"`` is a truthy string,
    and ``"max_upload_mb": "20480"`` makes the upload limit arithmetic raise.
    """
    bool_fields = [f for f in fields(config.AppSettings()) if f.type == "bool"]
    assert bool_fields, "expected at least one bool field; has the module changed?"

    loaded = config._apply(config.AppSettings(), {"keep_intermediate": "false"})
    assert loaded.keep_intermediate is False, \
        "the string 'false' is truthy - this would have kept every intermediate"

    loaded = config._apply(config.AppSettings(), {"max_upload_mb": "20480"})
    assert loaded.max_upload_mb == 20480
    assert isinstance(loaded.max_upload_mb, int)
    assert loaded.max_upload_mb * 1024 * 1024 > 0


@pytest.mark.parametrize("text,expected", [
    ("true", True), ("True", True), ("yes", True), ("on", True), ("1", True),
    ("false", False), ("no", False), ("off", False), ("0", False),
])
def test_boolean_spellings_a_user_might_type(text, expected):
    loaded = config._apply(config.AppSettings(), {"keep_intermediate": text})
    assert loaded.keep_intermediate is expected


def test_a_value_of_the_wrong_type_is_ignored_not_stored():
    """Better the default than a string that detonates later."""
    loaded = config._apply(config.AppSettings(), {"max_upload_mb": "not a number"})
    assert isinstance(loaded.max_upload_mb, int)


def test_unknown_keys_do_not_create_attributes():
    """A settings file from a newer build must not grow phantom fields."""
    loaded = config._apply(config.AppSettings(), {
        "not_a_setting": 1,
        "render": {"not_a_render_setting": 2},
    })
    assert not hasattr(loaded, "not_a_setting")
    assert not hasattr(loaded.render, "not_a_render_setting")


def test_a_partial_file_keeps_the_defaults_for_everything_else():
    """Real settings files are hand-edited and often partial."""
    loaded = config._apply(config.AppSettings(), {"render": {"track_talker": True}})
    assert loaded.render.track_talker is True
    assert loaded.render.layout == config.RenderSettings().layout
    assert loaded.asr.device == config.AsrSettings().device
