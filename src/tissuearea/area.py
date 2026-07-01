"""Physical tissue-area estimation (mm²) from a tissue mask + slide MPP.

A thumbnail-scale binary tissue mask maps linearly onto level 0, so each mask
pixel covers ``(W·mpp_x/mask_w) · (H·mpp_y/mask_h)`` µm², where ``W, H`` are the
level-0 slide dimensions. Two flavours are reported so callers can compare
against references of unknown semantics:

* ``whole_mm2``      — area of all detected tissue.
* ``largest_cc_mm2`` — area of the single largest connected component (one
  tissue section); plus ``top2_sum_mm2`` and the sorted ``section_areas_mm2``.

Connected components use 8-connectivity (``skimage.measure.label``).
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from skimage.measure import label, regionprops

from .config import MaskingConfig
from .masking import build_tissue_mask
from .slide import SlideReader, resolve_mpp_xy

__all__ = [
    "area_from_mask",
    "region_areas",
    "mask_from_thumbnail",
    "tissue_area_from_thumbnail",
    "tissue_area_for_slide",
    "resolve_mpp_xy",
]

# ``mask_from_thumbnail`` is the public name for the segmentation step; it is the
# same function as :func:`tissuearea.masking.build_tissue_mask`.
mask_from_thumbnail = build_tissue_mask


def area_from_mask(
    mask: np.ndarray,
    width: int,
    height: int,
    mpp_x: float,
    mpp_y: float,
) -> Dict[str, object]:
    """Convert a thumbnail-scale binary tissue mask to physical area (mm²).

    Args:
        mask: 2D boolean (or 0/1) tissue mask at thumbnail resolution.
        width, height: level-0 slide dimensions in pixels.
        mpp_x, mpp_y: level-0 microns per pixel.

    Returns:
        dict with ``whole_mm2``, ``largest_cc_mm2``, ``top2_sum_mm2``,
        ``n_sections``, ``section_areas_mm2`` (sorted desc), ``mask_fraction``,
        ``mask_w``, ``mask_h``.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    mask_h, mask_w = mask.shape
    per_px_um2 = (width * mpp_x / mask_w) * (height * mpp_y / mask_h)

    n_tissue = int(mask.sum())
    whole_mm2 = n_tissue * per_px_um2 / 1e6

    labels = label(mask, connectivity=2)
    # bincount index 0 is background; [1:] are per-component pixel counts.
    comp_counts = np.bincount(labels.ravel())[1:]
    section_areas = sorted(
        (float(c) * per_px_um2 / 1e6 for c in comp_counts), reverse=True
    )

    largest_cc_mm2 = section_areas[0] if section_areas else 0.0
    top2_sum_mm2 = float(sum(section_areas[:2])) if section_areas else 0.0

    return {
        "whole_mm2": float(whole_mm2),
        "largest_cc_mm2": float(largest_cc_mm2),
        "top2_sum_mm2": float(top2_sum_mm2),
        "n_sections": int(len(section_areas)),
        "section_areas_mm2": section_areas,
        "mask_fraction": float(n_tissue / mask.size) if mask.size else 0.0,
        "mask_w": int(mask_w),
        "mask_h": int(mask_h),
    }


def region_areas(
    mask: np.ndarray,
    width: int,
    height: int,
    mpp_x: float,
    mpp_y: float,
) -> List[Dict[str, object]]:
    """Per-connected-component tissue areas (mm²) with geometry, ranked desc.

    Args:
        mask: 2D boolean (or 0/1) tissue mask at thumbnail resolution.
        width, height: level-0 slide dimensions in pixels.
        mpp_x, mpp_y: level-0 microns per pixel.

    Returns:
        List of dicts (largest first), each with ``rank`` (1 = largest),
        ``label`` (component id in the label image), ``n_pixels``, ``area_mm2``,
        ``centroid_xy`` ``(x, y)`` in mask pixels, and ``bbox``
        ``(min_row, min_col, max_row, max_col)``. The ``area_mm2`` values match
        ``area_from_mask(...)["section_areas_mm2"]`` exactly.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")

    mask_h, mask_w = mask.shape
    per_px_um2 = (width * mpp_x / mask_w) * (height * mpp_y / mask_h)

    labels = label(mask, connectivity=2)
    regions: List[Dict[str, object]] = []
    for p in regionprops(labels):
        regions.append(
            {
                "label": int(p.label),
                "n_pixels": int(p.area),
                "area_mm2": float(p.area * per_px_um2 / 1e6),
                "centroid_xy": (float(p.centroid[1]), float(p.centroid[0])),
                "bbox": tuple(int(v) for v in p.bbox),
            }
        )
    regions.sort(key=lambda r: r["area_mm2"], reverse=True)
    for i, r in enumerate(regions, start=1):
        r["rank"] = i
    return regions


def tissue_area_from_thumbnail(
    thumbnail: np.ndarray,
    width: int,
    height: int,
    mpp_x: float,
    mpp_y: float,
    config: Optional[MaskingConfig] = None,
    *,
    labeled_output_path: Optional[Union[str, Path]] = None,
    label_min_area_mm2: float = 0.0,
) -> Dict[str, object]:
    """Segment ``thumbnail`` and return physical tissue-area candidates.

    If ``labeled_output_path`` is given, also render the thumbnail with each
    region outlined and area-labelled to that path (see
    :func:`tissuearea.draw.draw_region_labels`).
    """
    mask = build_tissue_mask(thumbnail, config=config)
    out = area_from_mask(mask, width, height, mpp_x, mpp_y)
    if labeled_output_path is not None:
        from .draw import draw_region_labels

        draw_region_labels(
            thumbnail,
            mask,
            width,
            height,
            mpp_x,
            mpp_y,
            output_path=str(labeled_output_path),
            min_area_mm2=label_min_area_mm2,
        )
    return out


def tissue_area_for_slide(
    path: Union[str, Path],
    config: Optional[MaskingConfig] = None,
    *,
    labeled_output_path: Optional[Union[str, Path]] = None,
    label_min_area_mm2: float = 0.0,
    mpp_fallback: Optional[float] = None,
) -> Dict[str, object]:
    """Open a slide (read-only) and compute its tissue-area candidates.

    Adds ``slide_id``, ``width``, ``height``, ``mpp_x``, ``mpp_y``, and
    ``mask_scale`` to the result returned by :func:`tissue_area_from_thumbnail`.
    When ``labeled_output_path`` is given, also save an annotated thumbnail with
    every region outlined and area-labelled. ``mpp_fallback`` (microns/pixel) is
    used when the slide itself carries no MPP or objective-power metadata.
    """
    cfg = config if config is not None else MaskingConfig()
    with SlideReader(path) as reader:
        width, height = reader.dimensions
        fallback = mpp_fallback if (mpp_fallback and mpp_fallback > 0) else reader.mpp
        mpp_x, mpp_y = resolve_mpp_xy(reader.properties, fallback_mpp=fallback)
        thumbnail = reader.get_thumbnail_array(cfg.mask_scale)
        out = tissue_area_from_thumbnail(
            thumbnail,
            width,
            height,
            mpp_x,
            mpp_y,
            config=cfg,
            labeled_output_path=labeled_output_path,
            label_min_area_mm2=label_min_area_mm2,
        )
        out.update(
            slide_id=reader.slide_id,
            width=int(width),
            height=int(height),
            mpp_x=float(mpp_x),
            mpp_y=float(mpp_y),
            mask_scale=int(cfg.mask_scale),
        )
        return out
