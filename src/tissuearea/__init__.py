"""tissuearea — physical tissue-area estimation (mm²) for whole slide images.

Combine the foundation-model-free EasyMSI tissue segmentation with a slide's
microns-per-pixel (MPP) metadata to estimate how much physical tissue (mm²) is
on a whole slide image, reporting both whole-slide and per-section (largest
connected component) area.

Quick start
-----------
>>> from tissuearea import tissue_area_for_slide, MaskingConfig
>>> out = tissue_area_for_slide("slide.svs", MaskingConfig(filter_grays=False))
>>> out["largest_cc_mm2"]
"""

from .area import (
    area_from_mask,
    mask_from_thumbnail,
    resolve_mpp_xy,
    tissue_area_for_slide,
    tissue_area_from_thumbnail,
)
from .config import MaskingConfig
from .masking import (
    build_tissue_mask,
    combine_masks,
    fill_small_holes,
    filter_grays,
    filter_pen_marks,
    otsu_mask,
    remove_small_objects,
    visualize_mask,
)
from .slide import SlideReader

__version__ = "0.1.0"

__all__ = [
    # area
    "area_from_mask",
    "tissue_area_from_thumbnail",
    "tissue_area_for_slide",
    "mask_from_thumbnail",
    "resolve_mpp_xy",
    # config
    "MaskingConfig",
    # masking
    "build_tissue_mask",
    "otsu_mask",
    "combine_masks",
    "filter_grays",
    "filter_pen_marks",
    "remove_small_objects",
    "fill_small_holes",
    "visualize_mask",
    # slide
    "SlideReader",
    # meta
    "__version__",
]
