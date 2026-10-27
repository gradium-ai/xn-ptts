"""Builds the README's artwork from the sources in docs/readme/src/.

    uvx --with pillow --with svg.path python docs/readme/build.py

The hero (hero.webp) is designed by hand and not built here. So is the footer banner
(footer.webp). This makes the divider and the\nlight and dark versions of the Winged Victory. GitHub shows an SVG in a README as an image, with no scripts,
external files or web fonts, so everything here is self-contained.
"""

from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageOps

HERE = Path(__file__).parent
SRC = HERE / "src"

GREY = "#7E7E7E"

def trimmed(path: Path) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    return img.crop(img.getbbox())


def divider() -> str:
    """A row of dots, grey enough to sit on both GitHub themes."""
    W, H, step = 1280, 16, 16
    dots = "".join(f'<circle cx="{x + step / 2}" cy="8" r="2"/>' for x in range(0, W, step))
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label=""><g fill="{GREY}" fill-opacity=".55">{dots}</g></svg>\n'


def logo_mask(svg: Path, size: int) -> Image.Image:
    """A 24x24 single-path icon (simple-icons) rasterized as a coverage mask.

    Each subpath is filled on its own and XORed in, which is even-odd filling: holes come out.
    """
    import re

    from svg.path import Move, parse_path

    d = re.search(r' d="([^"]+)"', svg.read_text()).group(1)
    mask = Image.new("1", (size, size), 0)
    scale = size / 24
    for sub in _subpaths(parse_path(d), Move):
        pts = []
        for seg in sub:
            for i in range(17):
                p = seg.point(i / 16)
                pts.append((p.real * scale, p.imag * scale))
        if len(pts) > 2:
            layer = Image.new("1", (size, size), 0)
            ImageDraw.Draw(layer).polygon(pts, fill=1)
            mask = ImageChops.logical_xor(mask, layer)
    return mask.convert("L")


def _subpaths(path, Move):
    sub = []
    for seg in path:
        if isinstance(seg, Move):
            if sub:
                yield sub
            sub = []
        else:
            sub.append(seg)
    if sub:
        yield sub


def drawn_mask(path: Path, size: int) -> Image.Image:
    """A white-on-transparent drawing (the dotted Hugging Face face) as a mask, squared up."""
    img = trimmed(path)
    side = max(img.size)
    square = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    square.paste(img, ((side - img.width) // 2, (side - img.height) // 2))
    lum = square.convert("L")
    alpha = square.getchannel("A")
    # The drawing is itself dots; blur them together so the grid samples its shape, not its holes.
    from PIL import ImageFilter

    mask = ImageChops.multiply(lum, alpha).filter(ImageFilter.GaussianBlur(side / 90))
    return ImageOps.autocontrast(mask.resize((size, size), Image.LANCZOS))


def globe_mask(size: int) -> Image.Image:
    """A globe: rim, equator, two parallels and a meridian, in the 24-unit simple-icons frame."""
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    u = size / 24
    w = round(1.9 * u)
    draw.ellipse((1 * u, 1 * u, 23 * u, 23 * u), outline=255, width=w)
    draw.ellipse((7 * u, 1 * u, 17 * u, 23 * u), outline=255, width=w)
    draw.line((1 * u, 12 * u, 23 * u, 12 * u), fill=255, width=w)
    draw.line((12 * u, 1 * u, 12 * u, 23 * u), fill=255, width=w)
    return mask


def dot_icon(mask: Image.Image, fill: str, n: int = 20) -> str:
    """A mask as an n x n grid of dots, each as opaque as the mask is covered under it."""
    small = mask.resize((n, n), Image.BOX)
    px = small.load()
    dots = []
    for y in range(n):
        for x in range(n):
            cov = px[x, y] / 255
            if cov > 0.12:
                op = "" if cov > 0.9 else f' fill-opacity="{cov:.2f}"'
                dots.append(f'<circle cx="{x + 0.5}" cy="{y + 0.5}" r=".4"{op}/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {n} {n}" width="48" height="48" role="img" aria-label=""><g fill="{fill}">{"".join(dots)}</g></svg>\n'


ICONS = {
    "gradium": lambda size: globe_mask(size),
    "huggingface": lambda size: drawn_mask(SRC / "huggingface.webp", size),
    "discord": lambda size: logo_mask(SRC / "icons" / "discord.svg", size),
}


def icons() -> None:
    for name, make in ICONS.items():
        mask = make(768)
        for theme, ink in THEME_INK.items():
            (HERE / f"icon-{name}-{theme}.svg").write_text(dot_icon(mask, ink))
    print(f"docs/readme/icon-*.svg  {len(ICONS) * 2} files")


# GitHub's text colours, so the icons read like the link text beside them in either theme.
THEME_INK = {"light": "#1F2328", "dark": "#F0F6FC"}


if __name__ == "__main__":
    root = HERE.parent.parent
    for name, make in {"divider.svg": divider}.items():
        out = HERE / name
        out.write_text(make())
        print(f"{out.relative_to(root)}  {out.stat().st_size / 1024:.0f} KB")
    icons()
