"""Process bootstrap that must run before any ML imports.

CTranslate2 (used by faster-whisper) loads CUDA DLLs through the plain Windows
loader, which does not honour ``os.add_dll_directory``. The NVIDIA wheels ship
``nvidia/*/bin`` folders that are not on PATH, so GPU inference fails with
"Library cublas64_12.dll is not found". Importing this module first fixes PATH
for the lifetime of the process.
"""

from __future__ import annotations

import glob
import os
import site
import sys
import sysconfig
from pathlib import Path

_bootstrapped = False


def _candidate_site_dirs() -> list[str]:
    dirs: list[str] = []
    for getter in (site.getsitepackages, site.getusersitepackages):
        try:
            value = getter()
        except Exception:
            continue
        if isinstance(value, str):
            dirs.append(value)
        elif value:
            dirs.extend(value)

    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            dirs.append(path)
    return [d for d in dirs if d]


def nvidia_bin_dirs() -> list[str]:
    """Locate installed NVIDIA CUDA runtime ``bin`` directories."""
    found: list[str] = []
    for base in _candidate_site_dirs():
        pattern = str(Path(base) / "nvidia" / "**" / "bin")
        found.extend(p for p in glob.glob(pattern, recursive=True) if p not in found)

    # torch >= 2.x bundles its own CUDA runtime DLLs.
    for base in _candidate_site_dirs():
        torch_lib = Path(base) / "torch" / "lib"
        if torch_lib.is_dir():
            found.append(str(torch_lib))
    return found


def setup_cuda() -> list[str]:
    """Put CUDA runtime DLL folders on PATH and in the DLL search path.

    Returns the directories that were registered. Safe to call repeatedly.
    """
    added: list[str] = []
    for path in nvidia_bin_dirs():
        if not Path(path).is_dir():
            continue
        if path.lower() not in os.environ.get("PATH", "").lower():
            os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        added.append(path)
        if hasattr(os, "add_dll_directory"):
            try:
                os.add_dll_directory(path)
            except OSError:
                pass
    return added


def ensure_ffmpeg_on_path() -> None:
    """Prefer a project-local FFmpeg build, else rely on the system PATH."""
    local = Path(__file__).resolve().parent.parent / "bin"
    if local.is_dir():
        os.environ["PATH"] = str(local) + os.pathsep + os.environ.get("PATH", "")


def bootstrap() -> None:
    global _bootstrapped
    if _bootstrapped:
        return
    ensure_ffmpeg_on_path()
    setup_cuda()
    if sys.platform == "win32":
        # Keep child console windows from flashing during long encodes.
        os.environ.setdefault("PYTHONUNBUFFERED", "1")
    _bootstrapped = True
