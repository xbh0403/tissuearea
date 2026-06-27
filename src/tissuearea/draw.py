"""Render a thumbnail with each tissue region outlined and area-labelled.

``draw_region_labels`` outlines every connected tissue component (its contour)
on a copy of the thumbnail and writes that region's physical area (mm²) at its
centroid, with a header summarising the region count and total/largest area.
Useful for eyeballing how the per-section areas map onto the slide.
"""

import os
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage.measure import label

from .area import region_areas

__all__ = ["draw_region_labels"]

RGB = Tuple[int, int, int]


def _put_text(
    img: np.ndarray,
    text: str,
    org: Tuple[int, int],
    scale: float,
    thickness: int,
    color: RGB,
    anchor: str = "tl",
) -> None:
    """Draw ``text`` with a filled black background box for legibility."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = org
    if anchor == "center":
        x -= tw // 2
        y += th // 2
    else:  # top-left of the text box
        y += th
    # Keep the label fully inside the image.
    x = int(max(0, min(x, img.shape[1] - tw - 1)))
    y = int(max(th + 1, min(y, img.shape[0] - baseline - 1)))
    cv2.rectangle(img, (x - 2, y - th - 2), (x + tw + 2, y + baseline + 2), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


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
    contour_color: RGB = (255, 0, 0),
    text_color: RGB = (255, 255, 0),
    contour_thickness: Optional[int] = None,
    font_scale: Optional[float] = None,
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
        contour_thickness, font_scale: override the size auto-scaled from the
            thumbnail dimensions.
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
        raise ValueError(
            f"Mask shape {mask.shape} must match thumbnail {img.shape[:2]}"
        )

    h_px, w_px = img.shape[:2]
    if contour_thickness is None:
        contour_thickness = max(1, round(min(h_px, w_px) / 400))
    if font_scale is None:
        font_scale = max(0.35, min(h_px, w_px) / 1200)
    font_thick = max(1, round(font_scale * 2))

    labels = label(mask, connectivity=2)
    regions = region_areas(mask, width, height, mpp_x, mpp_y)

    # Contours first, so text labels sit on top of every outline.
    for r in regions:
        comp = (labels == r["label"]).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, contour_color, contour_thickness)

    for r in regions:
        if r["area_mm2"] < min_area_mm2:
            continue
        cx, cy = r["centroid_xy"]
        text = f"#{r['rank']}: {r['area_mm2']:.1f} mm2" if show_index else f"{r['area_mm2']:.1f} mm2"
        _put_text(img, text, (int(cx), int(cy)), font_scale, font_thick, text_color, anchor="center")

    if show_header:
        if regions:
            total = sum(r["area_mm2"] for r in regions)
            header = (
                f"{len(regions)} regions | total {total:.1f} mm2 | "
                f"largest {regions[0]['area_mm2']:.1f} mm2"
            )
        else:
            header = "no tissue detected"
        _put_text(
            img, header, (5, 5), font_scale * 1.1, font_thick, (255, 255, 255), anchor="tl"
        )

    if output_path:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        Image.fromarray(img).save(output_path)

    return img
