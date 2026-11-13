"""Draw the folder icon for the ViralForge project folder.

The app icon is a play triangle, which is what any video tool uses. A folder
sitting next to other project folders needs to say what this one *does*: it takes
something long and wide and turns it into something short and vertical. So the
mark is a wide 16:9 bar becoming a tall 9:16 bar, on the same gradient tile as
the app icon so the two read as a pair.

A folder icon is looked at far more often at 16 and 32 pixels than at 256, so
this writes a contact sheet of the real sizes alongside the .ico: shapes that
only work at 256 are the usual way these end up as an unreadable smudge.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw, ImageFilter  # noqa: E402

# The interface's own colours, so the folder cannot drift from the product.
BACKDROP_TOP = (45, 225, 252)      # --accent
BACKDROP_BOTTOM = (124, 92, 255)   # --accent-2
INK = (7, 8, 12)                    # --bg
PAPER = (240, 252, 255)             # the "film" highlight, not pure white

SIZES = (16, 24, 32, 48, 64, 128, 256)
MASTER = 1024


def _rounded_mask(size: int, radius_ratio: float = 0.20) -> Image.Image:
    """Slightly tighter radius than the app tile: folders sit further away."""
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=int(size * radius_ratio), fill=255)
    return mask


def _gradient(size: int) -> Image.Image:
    base = Image.new("RGB", (size, size), BACKDROP_TOP)
    draw = ImageDraw.Draw(base)
    for y in range(size):
        t = y / max(1, size - 1)
        draw.line([(0, y), (size, y)],
                  fill=tuple(int(BACKDROP_TOP[i] + (BACKDROP_BOTTOM[i] - BACKDROP_TOP[i]) * t)
                             for i in range(3)))
    return base


def _tile(size: int) -> Image.Image:
    """Draw the mark at ``size`` natively.

    Downscaling a 1024px master is what makes folder icons turn to mush: the
    perforations and the arrow head survive at 256 and merge into a grey smear
    at 16. Drawing each size on its own terms, and dropping detail that cannot
    survive it, is the only way the small sizes stay readable. 16 and 24px are
    the sizes Explorer picks for a file list, so they get the simplest mark.
    """
    image = _gradient(size).convert("RGBA")

    # A soft highlight across the top, as on the app icon.
    gloss = Image.new("L", (size, size), 0)
    ImageDraw.Draw(gloss).ellipse(
        (-size * 0.35, -size * 0.80, size * 1.35, size * 0.40), fill=64)
    gloss = gloss.filter(ImageFilter.GaussianBlur(size * 0.08))
    sheen = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    sheen.putalpha(gloss)
    image = Image.alpha_composite(image, sheen)

    draw = ImageDraw.Draw(image)
    s = float(size)

    # The group is laid out as one block and centred, so the tile never looks
    # like a small drawing floating in a large empty square. Width is the
    # binding constraint - the two bars carry a 16:9 and a 9:16 aspect, so making
    # them taller means making them narrower - and the arrow gets the smallest
    # allowance that still reads as an arrow.
    wide_w, wide_h = 0.500, 0.281      # 16:9, the source
    tall_w, tall_h = 0.281, 0.500      # 9:16, the result
    gap, arrow_w = 0.040, 0.070
    group = wide_w + gap + arrow_w + gap + tall_w
    left = (1.0 - group) / 2.0
    mid = 0.5

    radius = max(1.0, s * 0.028)

    wx = s * left
    wy = s * (mid - wide_h / 2)
    draw.rounded_rectangle((wx, wy, wx + s * wide_w, wy + s * wide_h),
                           radius=radius, fill=PAPER)

    tx = s * (left + wide_w + gap + arrow_w + gap)
    ty = s * (mid - tall_h / 2)
    draw.rounded_rectangle((tx, ty, tx + s * tall_w, ty + s * tall_h),
                           radius=radius, fill=INK)

    # The arrow: the actual job. Stem plus a solid head, so it still reads as a
    # direction at 16px instead of as one dark smudge.
    ax0 = s * (left + wide_w + gap)
    ax1 = s * (left + wide_w + gap + arrow_w)
    ay = s * mid
    head = s * arrow_w * 0.70
    stem = max(1, int(round(s * 0.036)))
    draw.line([(ax0, ay), (ax1 - head * 0.85, ay)], fill=INK, width=stem)
    draw.polygon([(ax1, ay),
                  (ax1 - head, ay - head * 0.80),
                  (ax1 - head, ay + head * 0.80)], fill=INK)

    # Perforations only where they can be drawn as holes rather than as grey
    # dots. Below 48px they close up the tall bar and cost more than they add.
    if size >= 48:
        hole = max(2.0, s * 0.030)
        for row in range(3):
            cy = s * (mid - tall_h / 2) + s * tall_h * (0.24 + 0.26 * row)
            draw.ellipse((tx + s * tall_w * 0.5 - hole / 2, cy - hole / 2,
                          tx + s * tall_w * 0.5 + hole / 2, cy + hole / 2), fill=PAPER)

    clipped = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    clipped.paste(image, (0, 0), _rounded_mask(size))
    return clipped


def contact_sheet(sizes=SIZES) -> Image.Image:
    """The icon at each size Explorer will actually pick, on a folder-ish grey."""
    pad, gap = 20, 22
    width = pad * 2 + sum(s for s in sizes) + gap * (len(sizes) - 1)
    height = pad * 2 + max(sizes) + 46
    sheet = Image.new("RGB", (width, height), (32, 33, 36))
    draw = ImageDraw.Draw(sheet)
    x = pad
    for size in sizes:
        sheet.paste(_tile(size), (x, pad), _tile(size))
        draw.text((x, pad + max(sizes) + 10), f"{size}px", fill=(190, 190, 195))
        x += size + gap
    return sheet


def main() -> int:
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)

    # Each size is drawn at its own size, then assembled into one .ico, so
    # Explorer gets a real 16px icon rather than a shrunken 256px one.
    frames = [_tile(s) for s in SIZES]
    ico = ROOT / "clipforge-folder.ico"
    frames[-1].save(ico, format="ICO", sizes=[(s, s) for s in SIZES],
                    append_images=frames[:-1])
    frames[-1].resize((256, 256), Image.LANCZOS).save(assets / "clipforge-folder.png")

    sheet = ROOT / "data" / "cache" / "folder_icon_sizes.png"
    sheet.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet().save(sheet)

    for path in (ico, assets / "clipforge-folder.png", sheet):
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB")
    print(f"\n  review the sizes at: {sheet.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
