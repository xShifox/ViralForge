"""Widening a clip past the job's discovery window.

The bound that used to stop this was not a marketplace rule. It was the window
the job was searching in when it chose which clips to propose - and it was
enforced in three places, so removing one of them changed nothing and the failure
surfaced somewhere else entirely:

* the editor greyed out Save and told the user to shorten the window;
* the route marked the clip stale, which is the toast in the screenshot;
* the export stage then *skipped* the clip, so the render ran, wrote nothing,
  and came back "the render finished but no platform file could be written".

What actually has to be respected is each platform's own ceiling, and the
renderer trims to that before either of the other checks is reached.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from clipforge.api import routes   # noqa: E402
from clipforge.core import job_manager   # noqa: E402
from clipforge.pipeline import runner   # noqa: E402


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


# ------------------------------------------------------------ the rule itself

def test_the_editor_does_not_block_on_a_clip_longer_than_the_job_window():
    """Save has to be reachable, or the edit cannot even be submitted."""
    source = _read("web/src/components/ClipEditor.tsx")
    assert "tooShort || tooLong" not in source, (
        "the Save button is still disabled for a clip past the job's window")


def test_the_editor_stops_ordering_the_user_to_shorten_it():
    source = _read("web/src/components/ClipEditor.tsx")
    assert "shorten the window to re-render" not in source


def test_a_clip_too_short_is_still_refused():
    """The other half of the rule, and it is a real one: a 4-second clip cannot
    be rendered, and the renderer raises on it."""
    source = _read("web/src/components/ClipEditor.tsx")
    assert "disabled={saving || clip.state === 'rendering' || tooShort}" in source
    assert "minimum" in _read("clipforge/core/job_manager.py")


def _code_only(path: str) -> str:
    """The file with its comments removed.

    The old wording survives in the comment that explains what was removed, and a
    test that greps the whole file cannot tell documentation from a live rule.
    """
    out = []
    for line in _read(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith('"""') \
                or stripped.startswith("*"):
            continue
        out.append(line.split("  #", 1)[0])
    return "\n".join(out)


def test_the_route_no_longer_marks_a_long_clip_stale():
    """This is the message in the screenshot. A stale marker is what greys the
    clip out and blocks the re-render."""
    source = _code_only("clipforge/api/routes.py")
    assert "shorten the window to re-render" not in source
    assert "elif span > maximum + 0.05:" not in source, (
        "the route still marks a clip past the job window as stale")


def test_the_export_stage_no_longer_skips_a_long_clip():
    """The refusal that was invisible. It ran after the render, so the render
    produced nothing and the clip came back with 'no platform file could be
    written' - which reads like a disk or codec fault, not a length rule."""
    source = _code_only("clipforge/pipeline/runner.py")
    assert "outside {minimum:.0f}-{maximum:.0f}s; skipped" not in source
    assert "minimum - 0.05 <= span <= maximum + 0.05" not in source


def test_the_export_stage_still_refuses_a_clip_below_the_minimum():
    assert "if span < minimum - 0.05:" in _code_only("clipforge/pipeline/runner.py")


# --------------------------------------------------------------- one rule only

def test_the_length_rule_is_not_written_down_in_three_places():
    """It was. Three copies of one bound, and removing one changed nothing.

    Worth a test because the fix that actually worked was the last of the three;
    the first two could have been removed and the user would have seen exactly
    the same symptom.
    """
    offenders = []
    for path in ("clipforge/api/routes.py", "clipforge/core/job_manager.py",
                 "clipforge/pipeline/runner.py"):
        text = _read(path)
        if "maximum + 0.05" in text and "refus" not in text.lower():
            # a mention is fine; a *guard* is not. Look for the shape of a refusal.
            for line in text.splitlines():
                if "maximum + 0.05" in line and (
                        "return None" in line or "return False" in line
                        or "ClipState.stale" in line or "span >" in line):
                    offenders.append(f"{path}: {line.strip()[:70]}")
    assert not offenders, "length guards that block a widened clip:\n" + \
                         "\n".join(offenders)


# ------------------------------------------------- the bound that does belong

def test_the_discovery_stage_still_normalises_proposed_clips_to_the_band():
    """The one place the maximum is still a hard rule, and it has to stay.

    Two different questions were being answered by one number:

      * "which clips should I propose?" - a preference, and the job's own window
        is the right answer. This stage enforces it, and should.
      * "may the user widen one of them afterwards?" - not a preference question,
        and the job's window is no business answering it. This stage never runs
        on an edit, which is why it can keep the rule.

    The risk is that these look like the same bug. Someone reading
    ``enforce_clip_length`` and the export guard would reasonably conclude the
    whole bound is wrong and delete it, and then every job would start proposing
    clips longer than the window it was given. So: this one is load-bearing.
    """
    source = _code_only("clipforge/pipeline/runner.py")
    assert "if not (minimum - 0.05 <= duration <= maximum + 0.05):" in source, (
        "the discovery stage no longer normalises proposed clips to the band")
    assert "belt and braces: nothing leaves this stage outside the band" in \
        _read("clipforge/pipeline/runner.py")


def test_the_discovery_stage_runs_before_any_edit_not_after():
    """What makes the guard above safe: it is not on the edit path."""
    rerender = _read("clipforge/core/job_manager.py")
    assert "enforce_clip_length" not in rerender, (
        "a re-render must not run the discovery stage - that is what would trim "
        "the user's widened clip straight back to the job's window")


# ------------------------------------------------------------- the ceiling

def test_the_platform_ceiling_is_still_what_binds():
    """The bound that matters is the platform's, and it is enforced in the
    renderer - which is why the editor does not have to.

    Asserted on the line that clamps a plain clip, not on the substring
    ``preset.max_seconds``: the hook-trim line below it also mentions that name,
    so a substring check would still pass with the ceiling clamp removed.
    """
    source = _code_only("clipforge/render/renderer.py")
    assert "duration = max(1.0, min(clip.duration, preset.max_seconds))" in source, (
        "the renderer must still clamp a plain clip to the platform ceiling")
    assert "duration = duration" not in source, \
        "the clamp was replaced by a no-op"


def test_a_prepended_hook_still_comes_off_the_tail_first():
    """Separate from the clamp above: when a hook is prepended the clip gets that
    much less to play, so the ceiling applies to the two of them together."""
    source = _code_only("clipforge/render/renderer.py")
    assert "duration = min(duration, max(1.0, preset.max_seconds - hook_len))" \
        in source


def test_the_client_does_not_hold_its_own_copy_of_the_ceilings():
    """A table of platform ceilings in TypeScript is a second copy of a rule that
    lives in PLATFORM_PRESETS, free to drift, with the drift invisible."""
    source = _read("web/src/components/ClipEditor.tsx")
    assert "PLATFORM_CEILING" not in source, (
        "the client is duplicating the platform ceiling table")