"""The two things the interface asks the card.

One reports how much video memory is in use, and the other lists folders so the
output-folder step can be a dialog instead of a text field. Both exist because
the interface cannot get the information any other way: Windows will not tell a
process its own video memory, and a web page cannot be told which folder the
user picked.

Both of those limitations shape what is allowed to be reported. The memory route
must never present a guess as a reading, and the folder route must not hand back
a path that is not a folder or a listing the user cannot use.
"""
from __future__ import annotations

import os
import stat as stat_module
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipforge.api import routes
from clipforge.app import app

client = TestClient(app)


# ============================================================ video memory

@pytest.fixture(autouse=True)
def _clean_attribution():
    """The tracker is module state, so one test must not set another's baseline."""
    from clipforge import gpu

    gpu._attribution = gpu._Attribution()
    yield
    gpu._attribution = gpu._Attribution()


def test_the_reading_reports_the_card_totals(monkeypatch):
    from clipforge import gpu

    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 3960, 4228, 62))
    reading = gpu.memory()
    assert (reading.total_mb, reading.used_mb, reading.free_mb) == (8188, 3960, 4228)
    assert reading.utilization_pct == 62


def test_no_card_reads_as_unknown_rather_than_empty(monkeypatch):
    """A bar that shows zero because nothing answered is worse than no bar."""
    from clipforge import gpu

    monkeypatch.setattr(gpu, "_memory_query", lambda: (0, 0, 0, 0))
    reading = gpu.memory()
    assert reading.total_mb == 0
    assert reading.process_known is False


def test_an_idle_card_reports_no_share_yet(monkeypatch):
    """Before anything is held there is nothing to attribute, and saying 0 would
    be a claim rather than a measurement."""
    from clipforge import gpu

    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 101, 8087, 0))
    reading = gpu.memory()
    assert reading.baseline_mb == 101
    assert reading.process_known is False
    assert reading.process_mb is None


def test_the_share_is_the_rise_above_the_idle_level(monkeypatch):
    from clipforge import gpu

    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 101, 8087, 0))
    gpu.memory()                                     # sets the baseline while idle

    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 3960, 4228, 70))
    gpu.note_gpu_held(True)
    reading = gpu.memory()

    assert reading.process_known is True
    assert reading.process_mb == 3960 - 101
    assert reading.process_mb < reading.used_mb, \
        "the rest of the card belongs to other programs"


def test_releasing_clears_the_attribution(monkeypatch):
    from clipforge import gpu

    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 101, 8087, 0))
    gpu.memory()
    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 3960, 4228, 70))
    gpu.note_gpu_held(True)
    assert gpu.memory().process_known is True

    gpu.note_gpu_held(False)
    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 150, 8038, 0))
    after = gpu.memory()

    assert after.process_known is False, \
        "with nothing held there is no longer anything to attribute"
    assert after.used_mb == 150


def test_a_card_that_got_busier_while_held_is_not_reported_as_ours(monkeypatch):
    """The honest failure mode of measuring by difference.

    Another program opening a browser tab while a job runs pushes the reading up,
    and that difference is not ours. It is still reported - it is the only
    measurement available - which is why the interface says how it was arrived at
    rather than presenting it as a reading from the driver.
    """
    from clipforge import gpu

    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 100, 8088, 0))
    gpu.memory()
    gpu.note_gpu_held(True)
    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 4200, 3988, 80))
    reading = gpu.memory()
    assert reading.process_mb == 4100, "the whole rise, including what is not ours"
    assert reading.used_mb == 4200


def test_the_route_returns_the_reading(monkeypatch):
    from clipforge import gpu

    monkeypatch.setattr(gpu, "_memory_query", lambda: (8188, 3960, 4228, 62))
    body = client.get("/api/system/gpu").json()
    for key in ("total_mb", "used_mb", "free_mb", "utilization_pct",
                "process_mb", "process_known"):
        assert key in body, key
    assert body["total_mb"] == 8188


def test_the_route_survives_a_driver_that_will_not_answer(monkeypatch):
    from clipforge import gpu

    monkeypatch.setattr(gpu, "_memory_query", lambda: (0, 0, 0, 0))
    response = client.get("/api/system/gpu")
    assert response.status_code == 200
    assert response.json()["total_mb"] == 0


def test_unreadable_output_from_the_driver_is_not_reported_as_zero(monkeypatch):
    """A parse failure is a different thing from an empty card."""
    from clipforge import gpu

    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **k: type(
        "R", (), {"stdout": "not,a,number", "returncode": 0, "check": lambda s: None})())
    assert gpu._memory_query() == (0, 0, 0, 0)


# ================================================================= folders

def test_the_top_level_lists_drives_and_places():
    body = client.get("/api/system/folders").json()
    assert body["path"] == ""
    assert body["parent"] is None
    kinds = {f["kind"] for f in body["folders"]}
    assert "drive" in kinds, "the drives have to be reachable"
    names = {f["name"] for f in body["folders"]}
    for place in ("Desktop", "Documents", "Videos", "Downloads"):
        assert place in names, f"{place} is where people keep this"
    for entry in body["folders"]:
        assert Path(entry["path"]).is_dir(), entry["path"]


def test_listing_a_folder_returns_its_subfolders(tmp_path):
    (tmp_path / "Clips").mkdir()
    (tmp_path / "Other").mkdir()
    (tmp_path / "a-video.mp4").write_bytes(b"not a folder")

    body = client.get("/api/system/folders",
                      params={"path": str(tmp_path)}).json()

    assert body["path"] == str(tmp_path)
    assert {f["name"] for f in body["folders"]} == {"Clips", "Other"}, \
        "files must not appear: this chooses where output goes"
    assert body["parent"] == str(tmp_path.parent)


def test_a_file_is_not_a_folder(tmp_path):
    target = tmp_path / "clip.mp4"
    target.write_bytes(b"x")
    assert client.get("/api/system/folders",
                      params={"path": str(target)}).status_code == 404


def test_a_path_that_is_not_there_says_so(tmp_path):
    assert client.get("/api/system/folders",
                      params={"path": str(tmp_path / "nope")}).status_code == 404


def test_a_tilde_path_is_expanded(tmp_path, monkeypatch):
    """Typing ~\\Videos into the field has to work as typed.

    The environment variable is what is patched, not ``Path.home``: expansion
    goes through os.path.expanduser, which reads USERPROFILE on Windows, so
    patching the method that merely *reports* the home directory leaves the code
    under test reading the real one. That showed up as this test asserting
    against C:\\Users\\<the builder>\\Videos and quietly passing for the wrong
    reason on any machine with that folder.
    """
    home = tmp_path / "home"
    (home / "Videos").mkdir(parents=True)
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.setenv("HOME", str(home))

    body = client.get("/api/system/folders", params={"path": "~/Videos"}).json()
    assert body["path"] == str((home / "Videos").resolve())


def test_the_root_of_a_drive_has_no_parent():
    """'Up' from C:\\ has to be nowhere, not a loop back to itself."""
    body = client.get("/api/system/folders", params={"path": "C:\\"}).json()
    assert body["parent"] is None


def test_dot_folders_are_left_out(tmp_path):
    """A cache directory is not somewhere anyone chooses to put their clips."""
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "Visible").mkdir()
    body = client.get("/api/system/folders", params={"path": str(tmp_path)}).json()
    assert {f["name"] for f in body["folders"]} == {"Visible"}


def test_a_symlinked_folder_is_left_out(tmp_path):
    """The chooser cannot tell a link from its target, so it offers neither.

    This is the check that went wrong twice before getting right: the shell's
    own aliases come back with neither the hidden nor the system attribute set,
    and Path.stat() follows the link and so reports the *target's* attributes.
    Only lstat plus the reparse tag identifies them.
    """
    if os.name != "nt" and not hasattr(os, "symlink"):
        pytest.skip("no symlinks here")
    real = tmp_path / "RealFolder"
    real.mkdir()
    link = tmp_path / "LinkedFolder"
    try:
        os.symlink(real, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("this machine will not make a directory symlink")

    body = client.get("/api/system/folders", params={"path": str(tmp_path)}).json()
    names = {f["name"] for f in body["folders"]}
    assert "RealFolder" in names
    assert "LinkedFolder" not in names


def test_the_reparse_tags_are_the_ones_that_identify_a_link():
    """Guard the constants: picking the wrong flag filters nothing at all.

    0x800 is COMPRESSED, not REPARSE_POINT, and a mask built from it silently
    passes every junction through - which is exactly what it did.
    """
    assert stat_module.IO_REPARSE_TAG_MOUNT_POINT != 0x800
    assert stat_module.IO_REPARSE_TAG_SYMLINK != 0x800
    assert routes._is_hidden.__doc__


def test_an_empty_folder_is_not_an_error(tmp_path):
    body = client.get("/api/system/folders", params={"path": str(tmp_path)}).json()
    assert body["folders"] == []
