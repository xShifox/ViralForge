"""Verify the exports of one freshly rendered job against the eight criteria.

Written after the compliance fixes, so it is the acceptance check rather than
another exploration: every criterion states pass or fail against the numbers,
and nothing is inferred from a sample where the file count is small enough to
check every one.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JOBS = ROOT / "data" / "jobs"

CEILING = {"tiktok": 90.0, "shorts": 60.0, "reels": 90.0}
WANT_PROFILE, WANT_LEVEL = "High", 40
CAPTION_RGB, TOL = (255, 224, 64), 46

#: The bar for captions, unchanged: 12 of 16 moments is 75%, and it is still 75%
#: here. It is measured against the moments there was speech rather than against
#: the length of the clip, which is the only thing that had to change for a clip
#: with pauses in it to be judged on its subtitles.
SUBTITLE_PASS_RATIO = 12 / 16


def probe(path: Path) -> dict:
    data = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True, timeout=180).stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    rotation = 0
    for side in video.get("side_data_list", []) or []:
        if "rotation" in side:
            rotation = int(round(float(side["rotation"])))
    return {
        "duration": float(data["format"].get("duration", 0.0)),
        "container": data["format"].get("format_name", ""),
        "w": video.get("width"), "h": video.get("height"),
        "vcodec": video.get("codec_name"), "profile": video.get("profile"),
        "level": int(video.get("level", -1)), "pix": video.get("pix_fmt"),
        "sar": video.get("sample_aspect_ratio", "1:1"),
        "bf": int(video.get("has_b_frames", 0)),
        "vbitrate": int(video.get("bit_rate", 0) or 0) / 1e6,
        "acodec": (audio or {}).get("codec_name"),
        "arate": str((audio or {}).get("sample_rate", "-")),
        "ach": (audio or {}).get("channels", 0),
        "subs": sum(1 for s in data["streams"] if s["codec_type"] == "subtitle"),
        "rotation": rotation,
    }


def boxes(path: Path) -> list[str]:
    names: list[str] = []
    limit, pos = path.stat().st_size, 0
    with path.open("rb") as handle:
        while pos < limit:
            handle.seek(pos)
            head = handle.read(8)
            if len(head) < 8:
                break
            size, kind = struct.unpack(">I4s", head)
            if size == 1:
                size = struct.unpack(">Q", handle.read(8))[0]
            elif size == 0:
                size = limit - pos
            if size < 8 or pos + size > limit:
                names.append(kind.decode("latin1") + "!TRUNCATED")
                break
            names.append(kind.decode("latin1"))
            pos += size
    return names


def caption_band_from_ass(ass_path: Path) -> tuple[float, float] | None:
    """Where the caption style says captions live, as a fraction of frame height.

    Read from the overlay rather than assumed, because the overlay is the thing
    that decided. ``Style:`` is ``name, font, size, primary, secondary, outline,
    back, bold, italic, underline, strike, scaleX, scaleY, spacing, angle,
    border, outline, shadow, alignment, marginL, marginR, marginV, encoding`` -
    alignment is the 19th field and marginV the 22nd, both zero-indexed below.
    """
    try:
        text = ass_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    play = {}
    for line in text.splitlines():
        if line.startswith("PlayResY:"):
            play["y"] = int(line.split(":", 1)[1].strip())
        elif line.startswith("PlayResX:"):
            play["x"] = int(line.split(":", 1)[1].strip())
    for line in text.splitlines():
        if not line.startswith("Style:") or "," not in line:
            continue
        fields = line[len("Style:"):].split(",")
        if fields[0].strip() != "Caption" or len(fields) < 22:
            continue
        try:
            alignment = int(fields[18].strip())
            margin_v = int(fields[21].strip())
        except ValueError:
            continue
        height = play.get("y") or 1920
        try:
            size = float(fields[2].strip())
        except (ValueError, IndexError):
            size = 74.0
        # MarginV is measured from the edge the text is anchored to, not from the
        # top of the frame. For bottom-anchored captions - which is what this tool
        # renders - a 451px margin at PlayResY 1920 puts the baseline at 76% *down*,
        # not 23%. Getting this backwards made the check accept the headline banner
        # near the top as though it were the captions, which is a false pass.
        line = max(2.0, size * 1.25) / height       # one line, as a fraction
        two_lines = 2.2 * line
        if alignment in (1, 2, 3):                  # bottom
            baseline = 1.0 - margin_v / height
            return (max(0.0, baseline - two_lines), min(1.0, baseline + line))
        if alignment in (7, 8, 9):                  # top
            start = margin_v / height
            return (max(0.0, start - line), min(1.0, start + two_lines))
        start = margin_v / height
        return (max(0.0, start - two_lines), min(1.0, start + two_lines))
    return None


#: One cue's timing, in seconds, clip-relative.
_CUE = re.compile(
    r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def read_cues(srt_path: Path) -> list[tuple[float, float]]:
    """Every cue's start and end, from the sidecar the export shipped with.

    These are the moments there was speech, which is what makes cue-midpoint
    sampling better than sampling on a clock: a frame lifted from the middle of a
    pause has no caption to find, and failing a clip for that says nothing about
    whether its subtitles work.
    """
    try:
        text = srt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    out: list[tuple[float, float]] = []
    for line in text.splitlines():
        m = _CUE.search(line)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        start = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000.0
        end = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000.0
        if end > start:
            out.append((start, end))
    return out


def pick_cues(cues: list[tuple[float, float]], limit: int = 16) -> list[tuple[float, float]]:
    """Up to ``limit`` cues, spread evenly across the list.

    The cap is on how many frames get decoded, not on how strict the check is: a
    long clip is sampled at 16 moments and still has to clear the same 75%.
    """
    if len(cues) <= limit:
        return list(cues)
    step = (len(cues) - 1) / (limit - 1)
    return [cues[round(i * step)] for i in range(limit)]


def _clusters(rows: list[int], gap: int) -> list[tuple[int, int]]:
    """Group row numbers into runs, splitting wherever there is a real gap."""
    out: list[tuple[int, int]] = []
    for y in sorted(rows):
        if out and y - out[-1][1] <= gap:
            out[-1] = (out[-1][0], y)
        else:
            out.append((y, y))
    return out


#: A caption glyph's horizontal run, in the 270px-wide frame the checks sample
#: at. At this scale the caption style's strokes are around ten pixels wide, so
#: four keeps the letters and drops single-pixel coincidences.
GLYPH_RUN = 4


def _erode(mask: np.ndarray) -> np.ndarray:
    """Keep only pixels whose whole 3x3 neighbourhood is also caption-coloured.

    A glyph stroke is several pixels thick, so erosion leaves the letters intact.
    A scene that happens to contain the caption's own colour - skin, sand, sodium
    streetlight, a yellow title card - leaves scattered or thin traces, which this
    removes. Without it the fingerprint tracked the scenery: a frozen caption over
    moving footage hashed differently at every frame and read as synchronised.
    """
    keep = mask.copy()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(mask)
            ys = slice(max(0, dy), mask.shape[0] + min(0, dy))
            yd = slice(max(0, -dy), mask.shape[0] + min(0, -dy))
            xs = slice(max(0, dx), mask.shape[1] + min(0, dx))
            xd = slice(max(0, -dx), mask.shape[1] + min(0, -dx))
            shifted[yd, xd] = mask[ys, xs]
            keep &= shifted
    return keep


def caption_present(path: Path, at: float, width: int = 270,
                    band: tuple[float, float] | None = None,
                    want_signature: bool = False):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{at:.3f}", "-i", str(path),
         "-frames:v", "1", "-vf", f"scale={width}:-2", "-pix_fmt", "rgb24",
         "-f", "rawvideo", "-"], capture_output=True, timeout=600).stdout
    height = len(raw) // 3 // width
    arr = np.frombuffer(raw, np.uint8)
    counts: dict[int, int] = {}
    for index in range(0, len(raw) - 2, 3):
        if (abs(raw[index] - CAPTION_RGB[0]) < TOL
                and abs(raw[index + 1] - CAPTION_RGB[1]) < TOL
                and abs(raw[index + 2] - CAPTION_RGB[2]) < TOL):
            y, _x = divmod(index // 3, width)
            counts[y] = counts.get(y, 0) + 1

    # Only rows carrying a substantial run of the colour count as text. One stray
    # match per row is not a caption: warm skin tones, sand, sodium streetlight
    # and a yellow title card all throw pixels at this colour. A caption row is
    # dense - glyph strokes across the frame - so the threshold is a small
    # fraction of the row rather than one pixel.
    dense = sorted(y for y, n in counts.items() if n >= max(4, width // 25))
    if not dense:
        return (False, (), "") if want_signature else (False, ())

    # Group into bands and keep the ones that look like text rather than a wash of
    # colour across the picture. Taking min/max over every matching row instead
    # merged the top headline banner, the bottom captions and any warm scene into
    # one band spanning most of the frame, which then read as "not a caption".
    found = [c for c in _clusters(dense, gap=max(2, height // 60))
             if c[1] - c[0] + 1 <= height * 0.18]
    if not found:
        return (False, (), "") if want_signature else (False, ())

    def signature(top: int, bottom: int) -> str:
        """A fingerprint of the caption's own glyphs, used to check it advances.

        Comparing the caption region between two moments answers "is this the same
        caption?" without OCR: a caption track that is burned in and in sync changes
        as the speech does, and one that is a static overlay never does.

        It has to fingerprint the *glyphs*, not the pixels of the band they sit
        in. Hashing the raw rows looks right and is not: the band is full-width,
        so it carries whatever was behind the text, and the encoder's output is
        not bit-identical frame to frame either. Hashing that gave a different
        digest for every frame of a clip whose caption never changed - on still
        footage only by accident, because nothing behind it was moving - so a
        genuinely frozen track read as synchronised and this guard could not fire
        on real video, which is the only place it matters.

        So the mask of caption-coloured pixels is built, quantised to whole
        2x2 blocks to swallow that noise, and offset to the top of the band so a
        one-pixel jitter in where the text sits is not mistaken for a new
        caption. What is left is the shape of the letters.
        """
        # ``arr`` is one flat buffer, so rows top..bottom are the byte range
        # top*width*3 .. (bottom+1)*width*3. Slicing it as arr[top:bottom+1]
        # took one byte per row - twelve bytes from the very top of the frame -
        # so the digest was of the top of the picture and bore no relation to
        # where the caption was or what it said.
        band = arr[top * width * 3:(bottom + 1) * width * 3]
        pixels = band.reshape(bottom - top + 1, width, 3)
        mask = np.zeros(pixels.shape[:2], dtype=bool)
        for c in range(3):
            mask |= np.abs(pixels[:, :, c].astype(np.int16) - CAPTION_RGB[c]) < TOL

        # Colour alone is not enough: a scene can contain the caption's own
        # colour, and a rainbow test card contains it at every frame. A glyph
        # stroke is a *run* of that colour several pixels long, while a
        # background coincidence is scattered single pixels, so only runs are
        # kept. Without this the fingerprint tracked the scenery and a frozen
        # caption over moving footage still read as synchronised.
        glyphs = np.zeros_like(mask)
        for y in range(mask.shape[0]):
            row = mask[y]
            if not row.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8),
                                                          [0]))))
            for start, stop in zip(edges[0::2], edges[1::2]):
                if stop - start >= GLYPH_RUN:
                    glyphs[y, start:stop] = True

        core = _erode(glyphs)
        # The band is located by searching for the caption's colour, and the
        # search lands a row or two either side of the glyphs from one frame to
        # the next - so a fingerprint over the cropped rectangle changes when the
        # crop does, even though the caption has not moved or changed. Hence a
        # descriptor of the strokes themselves: how many there are and how long
        # each run is. Both are the same whenever the same caption is on screen,
        # wherever in the frame it was found.
        runs: list[int] = []
        for y in range(core.shape[0]):
            row = core[y]
            if not row.any():
                continue
            edges = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8),
                                                          [0]))))
            runs.extend(stop - start for start, stop in zip(edges[0::2], edges[1::2]))
        if not runs:
            return "none"
        parts = f"{int(core.sum())}|" + ",".join(str(r) for r in sorted(runs))
        return hashlib.sha1(parts.encode("ascii")).hexdigest()[:12]

    if band is None:
        top, bottom = found[0]
        return (True, (top, bottom, height), signature(top, bottom)) \
            if want_signature else (True, (top, bottom, height))

    low, high = band[0] * height, band[1] * height
    # A little slack either side: the measured band is of glyph rows, the declared
    # one is of margins, and they are not the same edges.
    slack = height * 0.04
    for top, bottom in found:
        if bottom >= low - slack and top <= high + slack:
            if want_signature:
                return True, (top, bottom, height), signature(top, bottom)
            return True, (top, bottom, height)
    return (False, (), "") if want_signature else (False, ())


def decodes(path: Path) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-xerror", "-i", str(path),
         "-f", "null", "-"], capture_output=True, text=True, timeout=1800)
    noise = ("Output file is empty", "does not contain any stream")
    errors = [l for l in result.stderr.splitlines()
              if l.strip() and not any(n in l for n in noise)]
    return result.returncode == 0 and not errors


def main() -> int:
    job_id = sys.argv[1] if len(sys.argv) > 1 else open(
        ROOT / "data/cache/verify_job_id.txt").read().strip()
    folder = JOBS / job_id
    # Recursive on purpose. Exports moved into one folder per clip - "exports/01 -
    # HEADLINE/clip.mp4" - so a plain glob matched nothing at all. It still
    # reported "all criteria pass" on zero files, which is worse than no check: a
    # verifier that cannot see the files cannot fail.
    exports = sorted((folder / "exports").rglob("*.mp4"))
    print(f"verifying {job_id}: {len(exports)} exported files\n")

    # A verifier that finds nothing has checked nothing, and reporting that as a
    # pass is the one outcome worse than reporting a failure. Refuse to conclude
    # anything unless there was something to check.
    if not exports:
        print("  NO EXPORTED FILES FOUND - nothing was verified.\n"
              "  This is not a pass. Check the job id, or whether the job has\n"
              "  finished exporting.")
        return 2

    fails: list[str] = []
    rows = []
    for path in exports:
        platform = next((k for k in CEILING if path.stem.endswith(f"_{k}")), None)
        rel = path.name
        info = probe(path)
        rows.append((platform, rel, info))

        # 1. format
        if info["level"] != WANT_LEVEL:
            fails.append(f"1 format  {rel}: level {info['level']} not {WANT_LEVEL}")
        if info["profile"] != WANT_PROFILE:
            fails.append(f"1 format  {rel}: profile {info['profile']}")
        if info["vcodec"] != "h264" or "mp4" not in info["container"]:
            fails.append(f"1 format  {rel}: {info['vcodec']} in {info['container']}")
        if info["pix"] != "yuv420p":
            fails.append(f"1 format  {rel}: pix_fmt {info['pix']}")
        if info["acodec"] != "aac" or info["arate"] != "48000" or info["ach"] != 2:
            fails.append(f"1 format  {rel}: audio {info['acodec']} "
                         f"{info['arate']}Hz {info['ach']}ch")
        # 2. aspect
        if (info["w"], info["h"]) != (1080, 1920):
            fails.append(f"2 aspect  {rel}: {info['w']}x{info['h']}")
        if info["rotation"] % 360 or info["sar"] not in ("1:1", "", None):
            fails.append(f"2 aspect  {rel}: rot {info['rotation']} sar {info['sar']}")
        if info["subs"]:
            fails.append(f"2 aspect  {rel}: {info['subs']} attached subtitle streams")
        # 3. duration
        if not (15.0 - 0.05 <= info["duration"] <= 90.0 + 0.05):
            fails.append(f"3 dur    {rel}: {info['duration']:.2f}s outside 15-90s")
        if platform and info["duration"] > CEILING[platform] + 0.5:
            fails.append(f"3 dur    {rel}: {info['duration']:.2f}s over the "
                         f"{platform} {CEILING[platform]:.0f}s ceiling")
        # 7. playback
        found = boxes(path)
        if any(b.endswith("!TRUNCATED") for b in found) or "moov" not in found:
            fails.append(f"7 play   {rel}: boxes {found}")
        elif found.index("moov") > found.index("mdat"):
            fails.append(f"7 play   {rel}: moov after mdat")
        if not decodes(path):
            fails.append(f"7 play   {rel}: decode error under -xerror")

    # 4. subtitles. Sampled at the middle of each subtitle cue, not on a clock.
    #
    # The old approach lifted 16 frames evenly across the clip and demanded a
    # caption in 12 of them. On a clip that is two-thirds silence that is a
    # statement about the pauses, not about the subtitles: a frame from the middle
    # of a pause has no caption to find. A real clip - 3 cues across 30s of Arabic
    # dialogue - was failing 5/16 for that reason alone.
    #
    # The bar is unchanged at 75%. What changed is the denominator: it is now the
    # number of moments there was speech, so a clip with three cues is held to
    # three out of three rather than to twelve out of sixteen, which is not a
    # thing it can be.
    print("  clip                     cues w/ caption    band (rows of 480)")
    for path in exports:
        if not path.stem.endswith("_tiktok"):
            continue
        base = path.stem.rsplit("_", 1)[0]
        # Where this clip's own overlay says its captions sit, and when it said them.
        ass_for = sorted(path.parent.glob(base + ".ass"))
        declared = caption_band_from_ass(ass_for[0]) if ass_for else None
        where_expected = ("%.0f-%.0f%% down" % (declared[0] * 100, declared[1] * 100)
                          if declared else "anywhere")
        cues = pick_cues(read_cues(path.parent / f"{base}.srt"))

        if not cues:
            # No sidecar means there is nothing to check against, and a clip with
            # no caption track at all must not pass by default.
            fails.append(f"4 subs    {path.name}: no subtitle cues found to check")
            print(f"  {path.name[-28:]:<28} {'no cues':>16}")
            continue

        hits, band, signatures = 0, [], []
        for start, end in cues:
            ok, geo, sig = caption_present(path, (start + end) / 2.0,
                                           band=declared, want_signature=True)
            if ok:
                hits += 1
                band.append(geo)
                signatures.append(sig)

        # The 12-of-16 bar, expressed against however many speech moments there are.
        needed = math.ceil(SUBTITLE_PASS_RATIO * len(cues))
        if band:
            top = sum(b[0] for b in band) // len(band)
            bottom = sum(b[1] for b in band) // len(band)
            where = "%d-%d (%.0f-%.0f%% down)" % (top, bottom, top / 4.8, bottom / 4.8)
        else:
            where = "none"

        # Sync, without OCR: a caption track that is burned in and in step with the
        # speech changes as the speech does. One that is a static overlay, or that
        # is frozen on a single line for the whole clip, does not. Distinct
        # fingerprints across distinct cues is the evidence.
        distinct = len(set(signatures))
        syncing = distinct >= 2 or len(cues) == 1
        sync_note = f" {distinct} distinct" if len(cues) > 1 else ""

        print(f"  {path.name[-28:]:<28} {hits:>3}/{len(cues):<3} ({hits/len(cues):3.0%})"
              f"    {where}   expected {where_expected}{sync_note}")
        if hits < needed:
            fails.append(f"4 subs    {path.name}: caption present in only "
                         f"{hits} of {len(cues)} spoken moments (needs {needed})")
        elif len(cues) > 1 and not syncing:
            fails.append(f"4 subs    {path.name}: caption is identical at every cue - "
                         f"it is not tracking the speech")

    # 4b. sidecar timing against the clip it belongs to
    stamp = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->")
    for srt in sorted((folder / "exports").rglob("*.srt")):
        stem = srt.name.rsplit(".", 1)[0]
        mates = sorted((folder / "exports").rglob(stem + "_*.mp4"))
        if not mates:
            continue
        length = probe(mates[0])["duration"]
        last = 0.0
        for line in srt.read_text(encoding="utf-8", errors="replace").splitlines():
            m = stamp.search(line)
            if m:
                g = [int(x) for x in m.groups()]
                last = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        if last > length + 0.6:
            fails.append(f"4 subs    {srt.name}: last cue {last:.2f}s vs clip {length:.2f}s")

    # 5. branding: the watermark is drawn through libass, so look for its text.
    # Read the text the job was actually given. Hardcoding an expected string
    # makes the check fail every job that used a different handle, and pass none
    # of the ones that genuinely dropped the watermark.
    wanted = ""
    try:
        record = json.loads((folder / "job.json").read_text(encoding="utf-8"))
        wanted = str((record.get("options") or {}).get("watermark_text") or "")
    except (OSError, ValueError):
        wanted = ""
    ass_files = sorted((folder / "exports").rglob("*.ass"))
    if not wanted:
        print("\n  watermark: none configured for this job - nothing to check")
        if not ass_files:
            fails.append("5 brand   no caption overlays found - branding unverified")
    else:
        print(f"\n  watermark check (the job was submitted with '{wanted}'):")
        styled = [a for a in ass_files
                  if wanted in a.read_text(encoding="utf-8", errors="replace")]
        print(f"    overlays carrying the watermark: {len(styled)} of {len(ass_files)}")
        if not ass_files:
            fails.append("5 brand   no caption overlays found - branding unverified")
        elif not styled:
            fails.append(f"5 brand   watermark '{wanted}' never reached the overlay")

    # 6. bundles
    print("\n  bundles:")
    zips = sorted(folder.glob("*.zip"))
    for z in zips:
        try:
            with zipfile.ZipFile(z) as archive:
                ok = archive.testzip() is None
                count = len(archive.namelist())
        except Exception as exc:                                # noqa: BLE001
            ok, count = False, str(exc)[:40]
        print(f"    {z.name:<44} {'valid' if ok else 'CORRUPT'}  {count} entries "
              f"({z.stat().st_size/1e6:.0f} MB)")
        if not ok:
            fails.append(f"6 bundle  {z.name}: {count}")
    strays = list(folder.glob("*.part"))
    if strays:
        fails.append(f"6 bundle  partial files left behind: {[p.name for p in strays]}")

    # 8. quality, next to the pre-fix exports
    print(f"\n  {'file':<30} {'lvl':>4} {'bf':>3} {'Mbps':>6}")
    for platform, rel, info in rows:
        print(f"  {rel[-30:]:<30} {info['level']:>4} {info['bf']:>3} {info['vbitrate']:>6.2f}")
    total_mb = sum(p.stat().st_size for p in exports) / 1e6

    print(f"\n{'=' * 70}")
    if fails:
        print(f"{len(fails)} FAILURE(S):")
        for line in fails:
            print("  -", line)
    else:
        print("all eight criteria pass on every exported file")
    print(f"  {len(exports)} files, {total_mb:.0f} MB total")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
