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
from typing import Dict, Optional, Union

import numpy as np
from skimage.measure import label

from .config import MaskingConfig
from .masking import build_tissue_mask
from .slide import SlideReader, resolve_mpp_xy

__all__ = [
    "area_from_mask",
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


def tissue_area_from_thumbnail(
    thumbnail: np.ndarray,
    width: int,
    height: int,
    mpp_x: float,
    mpp_y: float,
    config: Optional[MaskingConfig] = None,
) -> Dict[str, object]:
    """Segment ``thumbnail`` and return physical tissue-area candidates."""
    mask = build_tissue_mask(thumbnail, config=config)
    return area_from_mask(mask, width, height, mpp_x, mpp_y)


def tissue_area_for_slide(
    path: Union[str, Path],
    config: Optional[MaskingConfig] = None,
) -> Dict[str, object]:
    """Open a slide (read-only) and compute its tissue-area candidates.

    Adds ``slide_id``, ``width``, ``height``, ``mpp_x``, ``mpp_y``, and
    ``mask_scale`` to the result returned by :func:`tissue_area_from_thumbnail`.
    """
    cfg = config if config is not None else MaskingConfig()
    with SlideReader(path) as reader:
        width, height = reader.dimensions
        mpp_x, mpp_y = resolve_mpp_xy(reader.properties, fallback_mpp=reader.mpp)
        thumbnail = reader.get_thumbnail_array(cfg.mask_scale)
        out = tissue_area_from_thumbnail(thumbnail, width, height, mpp_x, mpp_y, config=cfg)
        out.update(
            slide_id=reader.slide_id,
            width=int(width),
            height=int(height),
            mpp_x=float(mpp_x),
            mpp_y=float(mpp_y),
            mask_scale=int(cfg.mask_scale),
        )
        return out
