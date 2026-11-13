"""Draw the ViralForge app icon.

Kept as a script rather than a committed binary so the icon is reproducible and
reviewable: it is built from the same colours the interface uses, so it cannot
drift away from the product's look.

Writes three things:
  assets/clipforge.ico   - the Windows window/taskbar icon, several sizes so it
                           stays sharp whichever one Explorer picks
  assets/clipforge.png   - a 512px master, for the favicon and anywhere else
  web/public/icon.png    - the favicon the browser tab shows
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from PIL import Image, ImageDraw  # noqa: E402

# The interface's own colours, so the icon and the product agree.
BACKDROP_TOP = (212, 175, 55)      # --accent (luxury gold)
BACKDROP_BOTTOM = (245, 197, 66)   # --accent-2 (lighter gold)
INK = (7, 8, 12)                    # --bg, the glyph is cut out in the dark
SHADOW = (4, 34, 43)

SIZES = (16, 24, 32, 48, 64, 128, 256)
MASTER = 1024


def _rounded_mask(size: int, radius_ratio: float = 0.235) -> Image.Image:
    """A rounded square, the shape used for the logo in the top bar."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    draw.rounded_rectangle(
        (0, 0, size - 1, size - 1),
        radius=int(size * radius_ratio),
        fill=255,
    )
    return mask


def _gradient(size: int) -> Image.Image:
    """A diagonal gradient, matching the button gradients in the interface."""
    base = Image.new("RGB", (size, size), BACKDROP_TOP)
    draw = ImageDraw.Draw(base)
    bottom = BACKDROP_BOTTOM
    for y in range(size):
        t = y / max(1, size - 1)
        draw.line(
            [(0, y), (size, y)],
            fill=tuple(
                int(BACKDROP_TOP[i] + (bottom[i] - BACKDROP_TOP[i]) * t) for i in range(3)
            ),
        )
    return base


def _play_glyph(draw: ImageDraw.ImageDraw, size: int) -> None:
    """A play triangle, cut into the tile.

    Drawn rather than pasted so it scales cleanly to 16px, where a separate
    bitmap would turn to mush. Left with sharp corners on purpose: rounding the
    point with a disc makes that disc wider than the triangle is at the vertex,
    which reads as a lollipop rather than a play button.
    """
    draw.polygon(
        [
            (size * 0.345, size * 0.275),
            (size * 0.715, size * 0.500),
            (size * 0.345, size * 0.725),
        ],
        fill=INK,
    )


def build(size: int = MASTER) -> Image.Image:
    image = _gradient(size).convert("RGBA")
    # A soft inner highlight at the top, so the tile reads as a lit surface
    # rather than a flat swatch.
    gloss = Image.new("L", (size, size), 0)
    gdraw = ImageDraw.Draw(gloss)
    gdraw.ellipse(
        (-size * 0.35, -size * 0.75, size * 1.35, size * 0.45),
        fill=70,
    )
    gloss = gloss.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(
        size * 0.08))
    white = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    white.putalpha(gloss)
    image = Image.alpha_composite(image, white)

    draw = ImageDraw.Draw(image)
    _play_glyph(draw, size)

    # Clip the gradient to the rounded square.
    tile = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    tile.paste(image, (0, 0), _rounded_mask(size))
    return tile


def main() -> int:
    master = build()

    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    public = ROOT / "web" / "public"
    public.mkdir(exist_ok=True)

    # Several sizes in one .ico so Explorer can pick whichever it wants and the
    # taskbar does not show a blurry 16px version.
    ico_sizes = [(s, s) for s in SIZES]
    master.resize((256, 256), Image.LANCZOS).save(
        assets / "clipforge.ico", format="ICO", sizes=ico_sizes)
    master.resize((512, 512), Image.LANCZOS).save(assets / "clipforge.png")
    master.resize((256, 256), Image.LANCZOS).save(public / "icon.png")

    for path in (assets / "clipforge.ico", assets / "clipforge.png", public / "icon.png"):
        print(f"  {path.relative_to(ROOT)}  {path.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
