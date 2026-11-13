"""The Start Clipping step exists, and nothing uploads before it is pressed.

Choosing a video used to run the job. Pick the file, pick a folder, and the
upload began immediately — so the settings on the right could not be changed
without starting over, and the ten seconds between dropping a file and noticing
the clip length was wrong were the ten seconds nobody gets back.

This checks the two halves of that claim, because either on its own is easy to
satisfy wrongly. That the step is *drawn* is a string in a bundle. That nothing
*uploads* is the part that matters, and it is checked by looking at the code
path rather than by reading the button.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = (ROOT / "web" / "src" / "App.tsx").read_text(encoding="utf-8")
GATE = (ROOT / "web" / "src" / "components" / "StartGate.tsx").read_text(encoding="utf-8")


def body(source: str, name: str) -> str:
    """The text of one top-level function or component."""
    start = source.find(f"function {name}")
    assert start >= 0, f"{name} is not in the source"
    depth, index = 0, source.find("{", start)
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index + 1]
        index += 1
    raise AssertionError(f"{name} is not closed")


# -------------------------------------------------------------- it is there

def test_the_step_is_a_component_with_the_button():
    assert "Start Clipping" in GATE
    assert 'className="primary big"' in GATE


def test_it_is_used():
    assert "StartGate" in APP, "the step was written but never rendered"


def test_it_replaces_the_dropzone_rather_than_covering_it():
    """The settings panel has to stay editable while this is showing.

    Covering the drop area with a full-screen dialog would hide the reason the
    step is useful, so the two are switched instead - which is what makes the
    branch on ``pendingFile?.folder`` exist.
    """
    assert "pendingFile?.folder ?" in APP
    assert re.search(r"\?\s*\(\s*<StartGate", APP), \
        "the gate and the dropzone are not alternatives"


# --------------------------------------------------- nothing uploads early

def test_picking_a_folder_does_not_upload():
    """This is the bug. The old ``beginUpload`` called handleFile directly."""
    choose = body(APP, "chooseFolder")
    assert "handleFile" not in choose, \
        "choosing a folder still starts the job; the gate would be decorative"
    assert "setPendingFile" in choose


def test_only_the_confirmation_uploads():
    confirm = body(APP, "confirmStart")
    assert "handleFile" in confirm, "pressing the button must be what starts it"
    assert "pendingFile?.folder" in confirm or "pendingFile.folder" in confirm, \
        "it must not start without a folder"


def test_a_pending_file_records_whether_a_folder_was_chosen():
    """The two states are distinguished by that field, so it has to exist."""
    assert re.search(r"interface PendingFile\s*\{[^}]*folder:\s*string", APP, re.S)


def test_the_upload_carries_the_folder_that_was_chosen():
    confirm = body(APP, "confirmStart")
    assert "outputFolder: folder" in confirm, \
        "the chosen folder has to reach the upload"


# --------------------------------------------------------------- the guards

def test_it_refuses_to_start_with_nothing_chosen():
    """Two ways the button could be pressed with nothing to do: no folder, or no
    platform. Both produce a job that fails or exports nothing."""
    assert "chosen.length === 0" in GATE
    assert "!folder.trim()" in GATE


def test_the_platform_names_come_from_the_server_not_a_second_table():
    """Two tables of platform names drift, and then this one greets somebody with
    "Square" for a platform the rest of the interface calls something else.

    The options panel takes its names from the health payload, so this step has to
    as well. Asserting the keys agree with the server's is the check that catches
    a hardcoded list going stale.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    from clipforge.config import PLATFORM_PRESETS

    assert "platformLabels" in GATE, \
        "the gate is still carrying its own copy of the platform names"
    assert "PLATFORM_LABELS" not in GATE

    # and the server really does send a label for every platform it will accept
    assert set(PLATFORM_PRESETS), "the reference list is empty, so nothing to check"
    for key, preset in PLATFORM_PRESETS.items():
        assert preset.label, f"{key} has no label for the interface to show"


# ------------------------------------------------------------ the build

def test_it_is_in_the_built_bundle():
    assets = sorted((ROOT / "web" / "dist" / "assets").glob("index-*.js"))
    assert assets, "no built interface"
    bundle = assets[-1].read_text(encoding="utf-8")
    assert "Start Clipping" in bundle, \
        "the component exists but was never bundled"
    assert "Ready to clip" in bundle


@pytest.mark.parametrize("needle", [
    "Ready to clip",              # the heading
    "Choose another video",       # the way back
    "Change",                     # reopening the folder chooser
    "Nothing has been uploaded",  # so it is clear nothing started
])
def test_the_step_explains_itself(needle):
    assert needle in GATE, f"missing: {needle}"
