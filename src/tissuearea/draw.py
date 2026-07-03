"""Render a thumbnail with each tissue region outlined and area-labelled.

``draw_region_labels`` outlines every connected tissue component (its contour)
on a copy of the thumbnail and writes that region's physical area (mm²) at its
centroid, with a header summarising the region count and total/largest area.

Contours are drawn with OpenCV; the text is rendered with a real anti-aliased
sans-serif font (Helvetica/Arial/DejaVu Sans, resolved at runtime) on
semi-transparent rounded label boxes, for a clean, publication-style look.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.measure import label

from .area import region_areas

__all__ = ["draw_region_labels"]

RGB = Tuple[int, int, int]

# Sans-serif candidates, most "scientific" first. macOS Helvetica/Arial, then
# Linux DejaVu/Liberation; bare names let fontconfig resolve. Resolved once.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "Arial.ttf",
    "Helvetica.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "DejaVuSans.ttf",
)
_resolved_font_path = None
_font_lookup_done = False


def _default_font_path() -> Optional[str]:
    """Find a sans-serif TrueType font once, caching the result."""
    global _resolved_font_path, _font_lookup_done
    if _font_lookup_done:
        return _resolved_font_path
    _font_lookup_done = True
    for cand in _FONT_CANDIDATES:
        try:
            ImageFont.truetype(cand, 12)
            _resolved_font_path = cand
            return cand
        except Exception:
            continue
    try:  # matplotlib bundles DejaVu Sans
        import matplotlib.font_manager as fm

        path = fm.findfont("DejaVu Sans")
        ImageFont.truetype(path, 12)
        _resolved_font_path = path
    except Exception:
        _resolved_font_path = None
    return _resolved_font_path


def _load_font(size: int, font_path: Optional[str] = None):
    path = font_path or _default_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:  # Pillow >= 10.1 returns a scalable default
        return ImageFont.load_default(size)
    except TypeError:  # pragma: no cover - very old Pillow
        return ImageFont.load_default()


def _draw_label(draw, text, xy, font, fill, anchor, img_wh, pad, radius):
    """Draw ``text`` on a semi-transparent rounded box, kept inside the image."""
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    tw, th = r - l, b - t
    x, y = xy
    if anchor == "center":
        x0, y0 = x - tw / 2 - pad, y - th / 2 - pad
    else:  # top-left
        x0, y0 = x - pad, y - pad
    w_px, h_px = img_wh
    box_w, box_h = tw + 2 * pad, th + 2 * pad
    x0 = max(0, min(x0, w_px - box_w))
    y0 = max(0, min(y0, h_px - box_h))
    draw.rounded_rectangle([x0, y0, x0 + box_w, y0 + box_h], radius=radius, fill=(0, 0, 0, 160))
    draw.text((x0 + pad - l, y0 + pad - t), text, font=font, fill=(*fill, 255))


def draw_region_labels(
    thumbnail: np.ndarray,
    mask: np.ndarray,
    width: int,
    height: int,
    mpp_x: float,
    mpp_y: float,
    *,
    output_path: Optional[str] = None,
    min_area_mm2: float = 0.0,
    contour_color: RGB = (220, 20, 30),
    text_color: RGB = (255, 255, 255),
    contour_thickness: Optional[int] = None,
    font_size: Optional[int] = None,
    font_path: Optional[str] = None,
    show_index: bool = True,
    show_header: bool = True,
) -> np.ndarray:
    """Outline and area-label each tissue region on the thumbnail.

    Args:
        thumbnail: RGB thumbnail (the one the ``mask`` was built from).
        mask: boolean tissue mask matching ``thumbnail[:2]``.
        width, height: level-0 slide dimensions in pixels.
        mpp_x, mpp_y: level-0 microns per pixel (for the area labels).
        output_path: if given, save the annotated PNG here.
        min_area_mm2: regions smaller than this still get a contour but no text
            label (use to de-clutter slides with many specks). ``0`` labels all.
        contour_color, text_color: RGB colors for outlines and area text.
        contour_thickness, font_size: override the size auto-scaled from the
            thumbnail dimensions.
        font_path: path to a specific ``.ttf``/``.ttc`` to use instead of the
            auto-resolved sans-serif.
        show_index: prefix each label with its rank (``#1`` = largest).
        show_header: draw a summary header (region count, total, largest area).

    Returns:
        Annotated RGB uint8 image (a copy; the input ``thumbnail`` is untouched).
    """
    img = np.array(thumbnail, dtype=np.uint8)  # copy; never mutate the caller's array
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = np.ascontiguousarray(img[:, :, :3])
    img = np.ascontiguousarray(img)

    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.shape != img.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} must match thumbnail {img.shape[:2]}")

    h_px, w_px = img.shape[:2]
    if contour_thickness is None:
        contour_thickness = max(1, round(min(h_px, w_px) / 400))
    if font_size is None:
        font_size = max(12, round(min(h_px, w_px) / 48))
    pad = max(2, font_size // 5)
    radius = max(2, pad)

    labels = label(mask, connectivity=2)
    regions = region_areas(mask, width, height, mpp_x, mpp_y)

    # 1) contours with OpenCV (drawn onto the RGB array).
    for r in regions:
        comp = (labels == r["label"]).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, contour_color, contour_thickness, lineType=cv2.LINE_AA)

    # 2) text with PIL on a single semi-transparent overlay (crisp sans-serif).
    base = Image.fromarray(img).convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    font = _load_font(font_size, font_path)
    header_font = _load_font(round(font_size * 1.12), font_path)

    for r in regions:
        if r["area_mm2"] < min_area_mm2:
            continue
        cx, cy = r["centroid_xy"]
        text = f"#{r['rank']}: {r['area_mm2']:.1f} mm²" if show_index else f"{r['area_mm2']:.1f} mm²"
        _draw_label(odraw, text, (cx, cy), font, text_color, "center", (w_px, h_px), pad, radius)

    if show_header:
        if regions:
            total = sum(r["area_mm2"] for r in regions)
            header = (
                f"{len(regions)} regions   total {total:.1f} mm²   "
                f"largest {regions[0]['area_mm2']:.1f} mm²"
            )
        else:
            header = "no tissue detected"
        _draw_label(odraw, header, (pad, pad), header_font, text_color, "tl", (w_px, h_px), pad, radius)

    out = np.array(Image.alpha_composite(base, overlay).convert("RGB"))

    if output_path:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        Image.fromarray(out).save(output_path)

    return out
