"""Configuration for tissue segmentation.

A self-contained ``MaskingConfig`` dataclass holding the parameters of the
Otsu/CLAHE + pen/gray-filter + morphology tissue-mask pipeline. The default
values reproduce the production EasyMSI segmentation exactly; construct a fresh
``MaskingConfig(**overrides)`` to request non-default behaviour (e.g.
``MaskingConfig(filter_grays=False)``), never mutate the defaults in place.
"""

from dataclasses import dataclass
from typing import Tuple

__all__ = ["MaskingConfig"]


@dataclass
class MaskingConfig:
    """Parameters for the tissue-mask pipeline (segmentation only).

    The defaults match the values used to validate against the reference areas
    and are treated as read-only: any non-default run should pass a new
    ``MaskingConfig`` rather than editing these.
    """

    # General
    mask_scale: int = 32                     # thumbnail downsampling factor (level-0 / scale)
    force_no_mask: bool = False              # if True, treat the whole thumbnail as tissue

    # Otsu / CLAHE pre-processing
    otsu_kernel_size: int = 3                # morphological-closing kernel after Otsu
    otsu_clip_limit: float = 2.0             # CLAHE clip limit
    otsu_tile_grid_size: Tuple[int, int] = (8, 8)  # CLAHE tile grid

    # Color filters (each toggles a step in the mask pipeline)
    filter_grays: bool = True                # drop near-neutral (low-chroma) pixels
    filter_red_pen: bool = True
    filter_blue_pen: bool = True
    filter_green_pen: bool = True
    filter_black_pen: bool = True

    # Small-object removal
    object_min_size_factor: float = 0.0001   # min object size as a fraction of thumbnail area
    remove_small_obj_avoid_overmask: bool = True
    remove_small_obj_overmask_thresh: int = 95

    # Hole filling
    hole_min_size_factor: float = 0.0001     # max hole size to fill, as a fraction of thumbnail area

    # Final dilation
    dilation_kernel_size: int = 5            # 0/1 disables the final dilation
