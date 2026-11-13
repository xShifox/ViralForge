"""Match the reference machine: fetch the model, then report what this machine is.

Two machines with the same settings still run at different speeds, and the
difference looks like a fault when it isn't. The graphics card is detected and
the speech model is run in whatever precision that card has room for: an RTX 4060
with 8 GB gets `cuda / float16`, a GTX 1650 with 4 GB gets `cuda / int8`. The
exported video is encoded to the same quality target either way, so the clips are
equivalent - only the time taken differs. The card's tier is printed so this is
visible rather than inferred.

What is *not* an adaptation is the graphics card itself. This build requires one,
for both speech recognition and video encoding, and says so rather than quietly
running on the CPU. So a machine reporting `cpu`, or `blocked`, or `libx264`, is
a genuine problem and is counted as one.

The thing that is genuinely missing on a fresh install is the 3 GB speech model,
which is otherwise downloaded in the middle of the first job, where it looks like
a hang.

So this does two things:

  --fetch   pre-downloads the speech model, so the first real job does not
            appear to freeze while 3 GB arrives
  (default) starts nothing and simply asks the running ViralForge what it can do
            and compares it with the reference

Run `--fetch` first, then start ViralForge and run it again to see the report.
"""
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

# What the reference machine uses. These are the *shape* of a healthy install,
# not a target: a smaller graphics card legitimately reports a different tier and
# a different speech precision, and that is the tool doing its job rather than a
# fault. The quality settings, though, are identical on every machine by design
# and are checked exactly.
REFERENCE = {
    "asr_model": "large-v3",
    "platforms": ["tiktok", "shorts", "reels"],
    "encoder_setting": "auto",
    "render_crf": 19,
    "output_moved_out": True,
    "require_gpu": True,
    "require_gpu_encoder": True,
}
# Values that must match regardless of hardware.
#
# ffmpeg is deliberately not in here. Matching on a vendor name only proves one
# particular build is installed: the ffmpeg build under test reports gyan.dev,
# but a BtbN or a manual build is equally good and would report a DIFF for no
# reason. So the check is "does ffmpeg run at all", and the build is printed as
# information only. Same reasoning as the graphics card printed further down.
#
# The last two are requirements rather than comparisons. There is no configuration
# of this build where speech on the CPU is correct, so checking it is not a
# difference to be reviewed - it is the thing that has to be true.
MUST_MATCH = ["ffmpeg_present", "asr_model", "platforms", "encoder_setting",
              "render_crf", "output_moved_out", "require_gpu",
              "require_gpu_encoder", "speech_on_gpu", "encoder_on_gpu"]


def fetch_model(data_dir: Path) -> int:
    """Download the speech model the way ViralForge will, without needing a GPU."""
    models = data_dir / "models"
    models.mkdir(parents=True, exist_ok=True)
    target = models / f"models--Systran--faster-whisper-{REFERENCE['asr_model']}"
    if target.is_dir() and any(target.rglob("*.bin")):
        size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
        print(f"  the speech model is already here ({size / 1e9:.1f} GB)")
        return 0

    print("  downloading the speech model, about 3 GB.")
    print("  This is the step that otherwise happens in the middle of the first job.")
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  [X] faster-whisper is not installed yet - run setup.cmd first")
        return 1
    try:
        WhisperModel(REFERENCE["asr_model"], device="cpu", compute_type="int8",
                     download_root=str(models), local_files_only=False)
    except Exception as exc:                                # noqa: BLE001
        print(f"  [X] the download failed: {exc}")
        return 1
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) if target.is_dir() else 0
    print(f"  done ({size / 1e9:.1f} GB)")
    return 0


def report(url: str, data_dir: Path) -> int:
    try:
        with urllib.request.urlopen(f"{url}/api/system/health", timeout=60) as response:
            health = json.loads(response.read())
    except Exception as exc:                                # noqa: BLE001
        print(f"  could not reach ViralForge at {url} ({exc})")
        print("  start it first:  python -m clipforge --open")
        return 1

    asr, encoder = health["asr"], health["encoder"]
    gpu = health.get("gpu") or {}
    version = health.get("ffmpeg_version", "")
    print("  what this machine reports")
    print("  " + "-" * 62)
    print(f"  {'ffmpeg':<26} {version.split(' ')[0]} {' '.join(version.split(' ')[1:3])}")
    print(f"  {'':<26} (any working build; the vendor name is not checked)")
    print(f"  {'speech model':<26} {REFERENCE['asr_model']}")
    print(f"  {'graphics card':<26} {gpu.get('name', 'none detected')}")
    print(f"  {'card memory':<26} {gpu.get('total_mb', 0)} MB")
    # The tier is the card's name for how hard it will be driven. Printed because
    # 'int8' next to the reference's 'float16' otherwise reads as a downgrade.
    print(f"  {'working mode':<26} {gpu.get('label', 'unknown')} ({gpu.get('tier', 'unknown')})")
    print(f"  {'speech device':<26} {asr['resolved_device']} / {asr['resolved_compute']}")
    print(f"  {'graphics cards':<26} {asr.get('cuda_devices', 0)}")
    print(f"  {'video encoder':<26} {encoder['active']}")
    print(f"  {'platforms':<26} {', '.join(health['platforms'])}")

    settings_file = data_dir / "settings.json"
    settings = {}
    if settings_file.is_file():
        settings = json.loads(settings_file.read_text(encoding="utf-8"))
    print(f"  {'render crf':<26} {settings.get('render', {}).get('crf')}")
    print(f"  {'encoder setting':<26} {settings.get('render', {}).get('encoder')}")
    print(f"  {'clips moved out':<26} {settings.get('move_exports_out')}")

    print("\n  checks")
    print("  " + "-" * 62)
    failures = 0
    values = {
        "ffmpeg_present": bool(health.get("ffmpeg")),
        "asr_model": settings.get("asr", {}).get("model") == REFERENCE["asr_model"],
        "platforms": settings.get("platforms") == REFERENCE["platforms"],
        "encoder_setting": settings.get("render", {}).get("encoder")
        == REFERENCE["encoder_setting"],
        "render_crf": settings.get("render", {}).get("crf") == REFERENCE["render_crf"],
        "output_moved_out": settings.get("move_exports_out") == REFERENCE["output_moved_out"],
        "require_gpu": settings.get("asr", {}).get("require_gpu")
        == REFERENCE["require_gpu"],
        "require_gpu_encoder": settings.get("render", {}).get("require_gpu_encoder")
        == REFERENCE["require_gpu_encoder"],
        # Both are requirements, not comparisons. This build does not run on the
        # CPU, so anything other than the card is a real fault - and one that
        # stops the job rather than slowing it down.
        "speech_on_gpu": asr["resolved_device"] == "cuda",
        "encoder_on_gpu": encoder["active"] == "h264_nvenc",
    }
    for key in MUST_MATCH:
        ok = values[key]
        failures += 0 if ok else 1
        print(f"  {'ok  ' if ok else 'DIFF'} {key}")
        if key == "speech_on_gpu" and not ok and asr.get("device_fallback"):
            print(f"       {asr['device_fallback']}")
        if key == "encoder_on_gpu" and not ok and encoder.get("fallback_reason"):
            print(f"       {encoder['fallback_reason'].splitlines()[0]}")

    # The remembered output folder is machine-specific by definition, so it can
    # never be compared against a reference. But a path left over from the
    # machine that built the bundle is worth flagging, because it points at a
    # drive the recipient does not have.
    remembered = settings.get("output_folder", "")
    if remembered and not Path(remembered).exists():
        print(f"  WARN remembered output folder does not exist here: {remembered}")
        print("       it is remembered only as the picker's starting point, so this")
        print("       is not a failure - but it is not this machine's folder either")
    elif remembered:
        print(f"  ok   remembered output folder exists: {remembered}")

    # HuggingFace's cache uses one flat "models--org--name" directory, not a
    # nested models--org / name pair. Getting this wrong makes the script report
    # a missing model on a machine that plainly has one.
    model = data_dir / "models" / f"models--Systran--faster-whisper-{REFERENCE['asr_model']}"
    if model.is_dir() and any(model.rglob("*.bin")):
        size = sum(p.stat().st_size for p in model.rglob("*") if p.is_file())
        print(f"  ok   speech model present ({size / 1e9:.1f} GB)")
    else:
        print("  MISS speech model - run this again with --fetch")
        failures += 1

    print("\n  differences that are expected, not faults")
    print("  " + "-" * 62)
    if gpu.get("tier") and gpu["tier"] != "none":
        print(f"  this card runs in \"{gpu['label']}\" mode.")
        print(f"  a different card runs in a different mode and takes a different")
        print(f"  amount of time. The speech model, resolution, subtitles and encoder")
        print(f"  quality are identical, so the exported clips are equivalent.")
        if gpu.get("tier") != "strong":
            print(f"  {gpu['name']} has {gpu['total_mb']} MB, so the speech model runs")
            print(f"  in {asr['resolved_compute']} rather than float16. That is enough")
            print(f"  memory being the deciding factor, not a fault.")
    elif not asr.get("cuda_devices"):
        print("  NO GRAPHICS CARD FOUND.")
        print("  This build requires an NVIDIA card for both speech recognition and")
        print("  video encoding, and will stop with an explanation rather than run on")
        print("  the CPU. The checks above will say so. Do not uninstall the NVIDIA")
        print("  CUDA packages; they are required, not optional.")
    if encoder.get("fallback_reason"):
        print(f"  encoding fell back because: {encoder['fallback_reason'].splitlines()[0]}")
    if asr.get("device_fallback"):
        print(f"  speech could not use the card because: {asr['device_fallback']}")

    print()
    print(f"  {'MATCHED' if not failures else f'{failures} difference(s) to look at'}")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Match the reference machine.")
    parser.add_argument("--fetch", action="store_true",
                        help="download the speech model, then stop")
    parser.add_argument("--url", default="http://127.0.0.1:8756",
                        help="a running ViralForge (default: %(default)s)")
    # Script-relative, not "data". A relative default only resolves when the
    # script happens to be run from the bundle root, and from anywhere else it
    # silently reads nothing - which showed up as five spurious DIFF lines and a
    # MISS speech model on a perfectly healthy install.
    parser.add_argument("--data", default=str(Path(__file__).resolve().parent / "data"),
                        help="the data folder (default: next to this script)")
    args = parser.parse_args()
    data_dir = Path(args.data)

    if args.fetch:
        print("  fetching the speech model")
        return fetch_model(data_dir)
    return report(args.url, data_dir)


if __name__ == "__main__":
    raise SystemExit(main())
