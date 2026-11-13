"""Where the chooser's "my Desktop" actually is.

The list was showing ``C:\\Users\\<you>\\Desktop`` while the Desktop the user sees
is ``C:\\Users\\<you>\\OneDrive\\Desktop``, which is where Windows puts it once
"Desktop" has been redirected. So the chooser opened on a leftover folder holding
five unrelated files, and every folder the user actually had was somewhere the
list never mentioned. It read as the chooser having lost their files.

Two things were wrong, and both had to be:

- The lookup asked for folder id ``0``, which is CSIDL_PROFILE - the home
  directory - and then ignored which folder had been asked for. So it answered
  "the profile" for Desktop, Documents, Videos and Downloads alike.
- Even once that was fixed, the profile path was tried first, and on this
  machine it exists. Preferring it is what sent the user to the wrong Desktop.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from clipforge.api import routes

WINDOWS_ONLY = pytest.mark.skipif(os.name != "nt", reason="Windows known folders")


# ------------------------------------------------------------- the lookup

@WINDOWS_ONLY
def test_it_answers_with_a_folder_that_exists():
    for name in routes.KNOWN_FOLDERS:
        found = routes._known_folder(name)
        if found is None:
            continue
        assert Path(found).is_dir(), f"{name} resolved to {found}, which is not there"


@WINDOWS_ONLY
def test_it_does_not_answer_with_the_profile_directory():
    """The bug: every name resolved to C:\\Users\\<you>.

    Anything under the profile passes this for Documents, which genuinely lives
    there, so the check is that it is never the profile itself and never ignores
    the name it was asked about.
    """
    profile = Path.home().resolve()
    for name in routes.KNOWN_FOLDERS:
        found = routes._known_folder(name)
        if not found:
            continue
        assert Path(found).resolve() != profile, \
            f"{name} resolved to the profile directory"


@WINDOWS_ONLY
def test_it_follows_a_redirected_desktop():
    """The case from the report, asserted only where it actually applies.

    Where the Desktop has been moved into OneDrive, the answer must be the moved
    location - not the leftover folder of the same name, which still exists and
    is not what the user sees.
    """
    desktop = routes._known_folder("Desktop")
    if not desktop:
        pytest.skip("this machine reports no Desktop")
    moved = Path(desktop)
    assert moved.is_dir()
    assert moved.name.lower() == "desktop", \
        "a known Desktop should still be called Desktop wherever it lives"
    # Where both exist, the one Windows reports is the redirected one.
    plain = Path.home() / "Desktop"
    if plain.is_dir() and plain.resolve() != moved.resolve():
        assert "onedrive" in str(moved).lower(), \
            f"expected the redirected Desktop, got {moved}"


@WINDOWS_ONLY
def test_an_unknown_name_is_refused_rather_than_guessed():
    assert routes._known_folder("Nowhere In Particular") is None
    assert routes._known_folder("") is None


def test_the_guids_are_shaped_like_guids():
    """A malformed identifier fails silently, which looks like no Desktop."""
    import uuid

    for name, text in routes.KNOWN_FOLDERS.items():
        parsed = uuid.UUID(text)          # raises on anything malformed
        assert parsed.version == 4, name


# ----------------------------------------------------------- the whole list

def test_the_top_level_offers_the_places_not_just_the_drives(tmp_path, monkeypatch):
    monkeypatch.setattr(routes.Path, "home", classmethod(lambda cls: tmp_path))
    for name in ("Desktop", "Videos"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(routes, "_known_folder", lambda name: None)

    body = routes.list_folders("")
    names = [f["name"] for f in body["folders"]]

    assert "Desktop" in names and "Videos" in names
    assert any(f["kind"] == "drive" or f["kind"] == "place" for f in body["folders"])
    for entry in body["folders"]:
        assert Path(entry["path"]).is_dir(), entry["path"]


def test_two_folders_with_one_name_are_both_offered(tmp_path, monkeypatch):
    """A redirected Desktop leaves two of them, and only one is the real one.

    Showing both, with the real one first, is honest: hiding the leftover would
    be tidier and would make the list disagree with what is on disk.
    """
    profile = tmp_path / "profile"
    (profile / "Desktop").mkdir(parents=True)
    real = tmp_path / "OneDrive" / "Desktop"
    real.mkdir(parents=True)

    monkeypatch.setattr(routes.Path, "home", classmethod(lambda cls: profile))
    monkeypatch.setattr(routes, "_known_folder",
                        lambda name: str(real) if name == "Desktop" else None)

    body = routes.list_folders("")
    desktops = [f for f in body["folders"] if f["name"] == "Desktop"]

    assert len(desktops) == 2, [f["path"] for f in desktops]
    assert desktops[0]["path"] == str(real), \
        "the folder Windows reports must come first"


def test_a_duplicated_folder_is_not_listed_twice(tmp_path, monkeypatch):
    """Where the two agree, there is one folder and it appears once."""
    home = tmp_path / "home"
    (home / "Videos").mkdir(parents=True)
    monkeypatch.setattr(routes.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(routes, "_known_folder",
                        lambda name: str(home / name) if name == "Videos" else None)

    body = routes.list_folders("")
    videos = [f for f in body["folders"] if f["name"] == "Videos"]

    assert len(videos) == 1


def test_a_missing_known_folder_is_simply_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(routes.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(routes, "_known_folder", lambda name: None)
    (tmp_path / "Desktop").mkdir()

    body = routes.list_folders("")
    names = [f["name"] for f in body["folders"]]

    assert "Desktop" in names
    assert "Videos" not in names, "a folder that is not there must not be offered"


# ------------------------------------------------- telling the two Desktops apart

def test_the_leftover_desktop_is_marked_as_the_leftover(tmp_path, monkeypatch):
    """Both Desktops are offered - that is deliberate - so they must be told apart.

    This is the failure that was actually reported: the chooser listed
    C:\\Users\\<you>\\Desktop first in the user's mind because the two rows are
    both called "Desktop" and differ only by a grey parent path, and the user's
    real folders - the ones with their finished clips in them - were in the
    OneDrive one that the list never made obviously preferable. Saving into the
    leftover looks like it worked and puts the clips somewhere they will not
    look.
    """
    profile = tmp_path / "profile"
    (profile / "Desktop").mkdir(parents=True)
    real = tmp_path / "OneDrive" / "Desktop"
    real.mkdir(parents=True)

    monkeypatch.setattr(routes.Path, "home", classmethod(lambda cls: profile))
    monkeypatch.setattr(routes, "_known_folder",
                        lambda name: str(real) if name == "Desktop" else None)

    body = routes.list_folders("")
    desktops = [f for f in body["folders"] if f["name"] == "Desktop"]

    assert len(desktops) == 2, [f["path"] for f in desktops]
    assert desktops[0]["path"] == str(real)
    assert desktops[0]["leftover"] is False, "the folder Windows reports is the real one"
    assert desktops[1]["path"] == str(profile / "Desktop")
    assert desktops[1]["leftover"] is True, \
        "the folder Windows did not name must be marked, or the two are the same row"


def test_a_sole_desktop_is_not_called_a_leftover(tmp_path, monkeypatch):
    """Nothing is a leftover when Windows named this one itself.

    The mark means "there is a better answer for this name", so a machine whose
    Desktop simply lives in the profile has no leftovers and must not be told it
    has one. Marking everything would be as misleading as marking nothing.
    """
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setattr(routes.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(routes, "_known_folder", lambda name: str(home / name))

    body = routes.list_folders("")
    desktops = [f for f in body["folders"] if f["name"] == "Desktop"]

    assert len(desktops) == 1
    assert desktops[0]["leftover"] is False


def test_a_desktop_offered_only_as_the_fallback_is_not_a_leftover(tmp_path, monkeypatch):
    """Windows answered nothing, so the profile folder is the best answer there is."""
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    monkeypatch.setattr(routes.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(routes, "_known_folder", lambda name: None)

    body = routes.list_folders("")
    desktops = [f for f in body["folders"] if f["name"] == "Desktop"]

    assert len(desktops) == 1
    assert desktops[0]["leftover"] is False


def test_every_entry_carries_the_mark_so_the_types_agree(tmp_path, monkeypatch):
    """The contract check compares the payload and the declared type field by field.

    A field present on the known places but missing from a drive or an ordinary
    folder is not a type error here - the check reads one real entry - it is a
    gate failure the next time it reads a different one.
    """
    home = tmp_path / "home"
    (home / "Desktop").mkdir(parents=True)
    (home / "Videos" / "nested").mkdir(parents=True)
    monkeypatch.setattr(routes.Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(routes, "_known_folder", lambda name: None)

    for body in (routes.list_folders(""), routes.list_folders(str(home / "Videos"))):
        for entry in body["folders"]:
            assert "leftover" in entry, entry
            assert isinstance(entry["leftover"], bool), entry
