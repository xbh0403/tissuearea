"""Tissue segmentation primitives and the full thumbnail -> mask pipeline.

These are vendored, dependency-free copies of the EasyMSI tissue-masking
primitives (Otsu/CLAHE, pen/gray color filters, small-object removal, hole
filling, final dilation). ``build_tissue_mask`` chains them in the exact order
the production pipeline uses, so a default-config mask is identical to the one
EasyMSI produces. The only inputs are an RGB thumbnail and a
:class:`~tissuearea.config.MaskingConfig`; there are no slide-, disk-, or
visualization side effects.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage import morphology

from .config import MaskingConfig

logger = logging.getLogger(__name__)

__all__ = [
    "otsu_mask",
    "filter_grays",
    "filter_pen_marks",
    "combine_masks",
    "remove_small_objects",
    "fill_small_holes",
    "visualize_mask",
    "build_tissue_mask",
    "DEFAULT_THRESHOLDS",
]


# ---------------------------------------------------------------------------
# Otsu / CLAHE base mask
# ---------------------------------------------------------------------------
def otsu_mask(
    thumbnail: np.ndarray,
    kernel_size: int = 3,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Foreground mask via CLAHE contrast enhancement + Otsu thresholding.

    Args:
        thumbnail: RGB uint8 thumbnail.
        kernel_size: morphological-closing kernel (``<=1`` disables closing).
        clip_limit: CLAHE clip limit.
        tile_grid_size: CLAHE tile grid.

    Returns:
        Boolean mask, ``True`` where tissue (darker than background) is detected.
    """
    if thumbnail.dtype != np.uint8:
        raise TypeError(f"Thumbnail must be uint8 for otsu_mask, got {thumbnail.dtype}")

    grayscale_img = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    equalized_img = clahe.apply(grayscale_img)
    # Invert so tissue (dark) becomes the high-intensity foreground for Otsu.
    img_inverted = 255 - equalized_img
    _thresh, threshold_img = cv2.threshold(
        img_inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        threshold_img = cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, kernel)
    return threshold_img.astype(bool)


# ---------------------------------------------------------------------------
# Color filters
# ---------------------------------------------------------------------------
def filter_grays(thumbnail: np.ndarray, tolerance: int = 15) -> np.ndarray:
    """Mask out near-neutral (low-chroma) pixels.

    Args:
        thumbnail: RGB uint8 thumbnail.
        tolerance: max absolute per-channel difference for a pixel to count as gray.

    Returns:
        Boolean mask, ``True`` where the pixel is NOT gray (i.e. keep it).
    """
    if thumbnail.dtype != np.uint8:
        raise TypeError(f"Thumbnail must be uint8 for filter_grays, got {thumbnail.dtype}")
    rgb_int = thumbnail.astype(int)
    rg_diff_ok = np.abs(rgb_int[:, :, 0] - rgb_int[:, :, 1]) <= tolerance
    rb_diff_ok = np.abs(rgb_int[:, :, 0] - rgb_int[:, :, 2]) <= tolerance
    gb_diff_ok = np.abs(rgb_int[:, :, 1] - rgb_int[:, :, 2]) <= tolerance
    is_gray = rg_diff_ok & rb_diff_ok & gb_diff_ok
    return ~is_gray


DEFAULT_THRESHOLDS: Dict[str, List[Tuple[int, int, int]]] = {
    "red": [
        (150, 80, 90), (110, 20, 30), (185, 65, 105), (195, 85, 125),
        (220, 115, 145), (125, 40, 70), (100, 50, 65), (85, 25, 45),
    ],
    "blue": [
        (60, 120, 190), (120, 170, 200), (175, 210, 230), (145, 180, 210),
        (37, 95, 160), (30, 65, 130), (130, 155, 180), (40, 35, 85),
        (30, 20, 65), (60, 60, 120), (110, 110, 175),
    ],
    "green": [
        (150, 160, 140), (70, 110, 110), (45, 115, 100), (30, 75, 60),
        (195, 220, 210), (225, 230, 225), (170, 210, 200), (20, 30, 20),
        (50, 60, 40), (30, 50, 35), (65, 70, 60), (100, 110, 105),
        (165, 180, 180), (140, 140, 150), (185, 195, 195),
    ],
    "black": [
        (50, 50, 50), (30, 30, 30), (20, 20, 20), (10, 10, 10),
    ],
}


def _base_filter(
    img_array: np.ndarray,
    thresholds: List[Tuple[int, int, int]],
    mode: str,
    dilate_kernel_size: Optional[int] = 3,
) -> np.ndarray:
    """RGB-threshold pen detector. Returns a boolean mask of detected pen pixels."""
    combined_mask = np.zeros(img_array.shape[:2], dtype=bool)
    r_channel, g_channel, b_channel = (
        img_array[:, :, 0],
        img_array[:, :, 1],
        img_array[:, :, 2],
    )

    for T in thresholds:
        if mode == "blue":      # R < T0, G < T1, B > T2
            r_mask = r_channel < T[0]
            g_mask = g_channel < T[1]
            b_mask = b_channel > T[2]
        elif mode == "red":     # R > T0, G < T1, B < T2
            r_mask = r_channel > T[0]
            g_mask = g_channel < T[1]
            b_mask = b_channel < T[2]
        elif mode == "green":   # R < T0, G > T1, B > T2
            r_mask = r_channel < T[0]
            g_mask = g_channel > T[1]
            b_mask = b_channel > T[2]
        elif mode == "black":   # R < T0, G < T1, B < T2
            r_mask = r_channel < T[0]
            g_mask = g_channel < T[1]
            b_mask = b_channel < T[2]
        else:
            raise ValueError(f"Unknown filter mode: {mode}")

        combined_mask |= r_mask & g_mask & b_mask

        # If the mask explodes, it is capturing background/error — stop.
        if np.count_nonzero(combined_mask) > 0.8 * combined_mask.size:
            break

    if dilate_kernel_size and dilate_kernel_size > 0:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
        return cv2.dilate(combined_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return combined_mask


def filter_pen_marks(
    thumbnail: np.ndarray,
    color: str = "red",
    thresholds: Optional[List[Tuple[int, int, int]]] = None,
    dilate: bool = True,
    base_otsu_mask: Optional[np.ndarray] = None,
    pen_percentage_threshold: float = 0.2,
) -> np.ndarray:
    """Detect pen marks of a given color and return the *keep* mask.

    Args:
        thumbnail: RGB uint8 thumbnail.
        color: one of ``red``, ``blue``, ``green``, ``black``.
        thresholds: custom RGB thresholds; defaults to :data:`DEFAULT_THRESHOLDS`.
        dilate: dilate detected pen regions to fully cover marks.
        base_otsu_mask: if given, used to veto the filter when pen "covers" too
            much tissue (assumed a false positive).
        pen_percentage_threshold: veto fraction of tissue area.

    Returns:
        Boolean mask, ``True`` where the pixel is NOT pen (i.e. keep it).
    """
    color = color.lower()
    if color not in DEFAULT_THRESHOLDS:
        raise ValueError(
            f"Unsupported color '{color}'. Choose from {list(DEFAULT_THRESHOLDS)}"
        )
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS[color]

    if not dilate:
        dilate_kernel_size = 0
    elif color == "blue":
        dilate_kernel_size = 6
    else:
        dilate_kernel_size = 3

    pen_mask = _base_filter(thumbnail, thresholds, color, dilate_kernel_size)

    if base_otsu_mask is not None and base_otsu_mask.any():
        otsu_area = np.count_nonzero(base_otsu_mask)
        pen_area_on_tissue = np.count_nonzero(pen_mask & base_otsu_mask)
        if otsu_area > 0 and (pen_area_on_tissue / otsu_area) > pen_percentage_threshold:
            logger.warning(
                "Detected %s pen marks cover > %.0f%% of tissue; ignoring %s filter.",
                color, pen_percentage_threshold * 100, color,
            )
            return np.ones_like(pen_mask, dtype=bool)

    return ~pen_mask


def combine_masks(
    thumbnail: np.ndarray,
    filters: Optional[List[str]] = None,
    base_otsu_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Intersect the base Otsu mask with the requested color filters.

    Args:
        thumbnail: RGB uint8 thumbnail.
        filters: subset of ``{red_pen, blue_pen, green_pen, black_pen, filter_grays}``.
        base_otsu_mask: precomputed Otsu mask; computed with defaults if ``None``.

    Returns:
        Boolean combined keep-mask.
    """
    if base_otsu_mask is None:
        base_otsu_mask = otsu_mask(thumbnail)
    if not filters:
        return base_otsu_mask.astype(bool)

    filter_functions: Dict[str, Any] = {
        "red_pen": lambda: filter_pen_marks(thumbnail, "red", base_otsu_mask=base_otsu_mask),
        "blue_pen": lambda: filter_pen_marks(thumbnail, "blue", base_otsu_mask=base_otsu_mask),
        "green_pen": lambda: filter_pen_marks(thumbnail, "green", base_otsu_mask=base_otsu_mask),
        "black_pen": lambda: filter_pen_marks(thumbnail, "black", base_otsu_mask=base_otsu_mask),
        "filter_grays": lambda: filter_grays(thumbnail),
    }

    combined_mask = base_otsu_mask.astype(bool)
    for filter_name in filters:
        if filter_name not in filter_functions:
            logger.warning("Unknown filter '%s' requested in combine_masks.", filter_name)
            continue
        try:
            combined_mask = np.logical_and(combined_mask, filter_functions[filter_name]())
        except Exception as e:  # noqa: BLE001 - a bad filter shouldn't kill the mask
            logger.warning("Failed to apply filter '%s': %s", filter_name, e)
    return combined_mask


# ---------------------------------------------------------------------------
# Morphology
# ---------------------------------------------------------------------------
# scikit-image >= 0.26 uses ``max_size`` (removes objects/holes of size <=
# max_size) in place of the old ``min_size``/``area_threshold`` (which removed
# size < threshold). So ``max_size = threshold - 1`` reproduces the prior result.
def remove_small_objects(
    mask: np.ndarray,
    min_size: Optional[float] = None,
    avoid_overmask: bool = True,
    overmask_thresh: int = 95,
    connectivity: int = 1,
) -> np.ndarray:
    """Drop foreground specks smaller than ``min_size`` pixels.

    When ``avoid_overmask`` is set and the result still covers ``>=
    overmask_thresh``% of the image, ``min_size`` is halved repeatedly (down to
    1) to back off an over-aggressive removal.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.size == 0:
        return mask

    if min_size is None:
        min_size = max(1, int(mask.size * 0.0001))
    else:
        min_size = max(1, int(min_size))

    cleaned_mask = morphology.remove_small_objects(
        mask, max_size=min_size - 1, connectivity=connectivity
    )

    if avoid_overmask and cleaned_mask.any():
        mask_percentage = (np.sum(cleaned_mask) / cleaned_mask.size) * 100
        current_min_size = min_size
        while mask_percentage >= overmask_thresh and current_min_size > 1:
            current_min_size = max(1, current_min_size // 2)
            cleaned_mask = morphology.remove_small_objects(
                mask, max_size=current_min_size - 1, connectivity=connectivity
            )
            mask_percentage = (np.sum(cleaned_mask) / cleaned_mask.size) * 100
            if current_min_size == 1:
                break

    return cleaned_mask


def fill_small_holes(
    mask: np.ndarray,
    min_size: Optional[float] = None,
    connectivity: int = 1,
) -> np.ndarray:
    """Fill background holes smaller than ``min_size`` pixels inside tissue."""
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if min_size is None:
        min_size = max(1, int(mask.size * 0.0001))
    return morphology.remove_small_holes(
        mask, max_size=int(min_size) - 1, connectivity=connectivity
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def visualize_mask(
    thumbnail: np.ndarray,
    mask: np.ndarray,
    output_path: Optional[str] = None,
) -> np.ndarray:
    """Black out non-tissue regions of ``thumbnail`` for visual QA.

    Args:
        thumbnail: RGB uint8 thumbnail.
        mask: boolean mask matching ``thumbnail[:2]``; ``True`` is kept.
        output_path: optional PNG path to save the visualization.

    Returns:
        RGB uint8 image with background blacked out.
    """
    if thumbnail.dtype != np.uint8:
        raise TypeError(f"Thumbnail must be uint8 for visualize_mask, got {thumbnail.dtype}")
    if thumbnail.ndim != 3 or thumbnail.shape[2] != 3:
        raise ValueError("Thumbnail must be RGB for visualize_mask")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask.shape != thumbnail.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} must match thumbnail {thumbnail.shape[:2]}"
        )

    vis_image = np.copy(thumbnail)
    vis_image[~mask] = (0, 0, 0)

    if output_path:
        dir_name = os.path.dirname(output_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        Image.fromarray(vis_image).save(output_path)
    return vis_image


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def build_tissue_mask(
    thumbnail: np.ndarray,
    config: Optional[MaskingConfig] = None,
) -> np.ndarray:
    """Build the binary tissue mask from an RGB thumbnail.

    Chains the primitives in the production order — ``otsu_mask`` ->
    ``combine_masks`` (color filters) -> ``remove_small_objects`` ->
    ``fill_small_holes`` -> optional dilation — driven by ``config``. With the
    default :class:`~tissuearea.config.MaskingConfig` the result is identical to
    the EasyMSI production segmentation.

    Args:
        thumbnail: RGB thumbnail (coerced to uint8 if needed).
        config: masking parameters; a fresh default is used when ``None``.

    Returns:
        Boolean 2D tissue mask at thumbnail resolution.
    """
    cfg = config if config is not None else MaskingConfig()

    if thumbnail.dtype != np.uint8:
        thumbnail = thumbnail.astype(np.uint8)

    if cfg.force_no_mask:
        return np.ones(thumbnail.shape[:2], dtype=bool)

    filters: List[str] = []
    if cfg.filter_red_pen:
        filters.append("red_pen")
    if cfg.filter_blue_pen:
        filters.append("blue_pen")
    if cfg.filter_green_pen:
        filters.append("green_pen")
    if cfg.filter_black_pen:
        filters.append("black_pen")
    if cfg.filter_grays:
        filters.append("filter_grays")

    thumb_size = thumbnail.shape[0] * thumbnail.shape[1]
    remove_min = max(1, int(thumb_size * cfg.object_min_size_factor))
    fill_min = max(1, int(thumb_size * cfg.hole_min_size_factor))

    base = otsu_mask(
        thumbnail,
        kernel_size=cfg.otsu_kernel_size,
        clip_limit=cfg.otsu_clip_limit,
        tile_grid_size=cfg.otsu_tile_grid_size,
    )
    combined = combine_masks(thumbnail=thumbnail, filters=filters, base_otsu_mask=base)
    cleaned = remove_small_objects(
        mask=combined,
        min_size=remove_min,
        avoid_overmask=cfg.remove_small_obj_avoid_overmask,
        overmask_thresh=cfg.remove_small_obj_overmask_thresh,
    )
    final_mask = fill_small_holes(mask=cleaned, min_size=fill_min)

    if cfg.dilation_kernel_size > 1:
        kernel = np.ones((cfg.dilation_kernel_size, cfg.dilation_kernel_size), np.uint8)
        final_mask = cv2.dilate(final_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    return final_mask.astype(bool)
