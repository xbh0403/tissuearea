"""Tests for the vendored tissue-mask pipeline and primitives."""

import warnings

import numpy as np
import pytest
from skimage.measure import label

from tissuearea import MaskingConfig, build_tissue_mask, visualize_mask
from tissuearea.masking import fill_small_holes, filter_grays, otsu_mask, remove_small_objects


def _synthetic_thumbnail(seed: int = 0) -> np.ndarray:
    """Pinkish H&E-ish blob (+ a smaller one) on a near-white field."""
    rng = np.random.default_rng(seed)
    thumb = np.full((120, 200, 3), 235, dtype=np.uint8)
    thumb[30:90, 20:110] = np.array([180, 110, 170], dtype=np.uint8)
    thumb[60:80, 150:180] = np.array([170, 100, 160], dtype=np.uint8)
    noise = rng.integers(-12, 12, size=thumb.shape, dtype=np.int16)
    return np.clip(thumb.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_build_tissue_mask_shape_and_dtype():
    thumb = _synthetic_thumbnail()
    mask = build_tissue_mask(thumb)
    assert mask.dtype == bool
    assert mask.shape == thumb.shape[:2]


def test_build_tissue_mask_detects_blob_not_background():
    thumb = _synthetic_thumbnail()
    mask = build_tissue_mask(thumb)
    # The colored blob center should be tissue; a near-white corner should not.
    assert mask[60, 60]  # inside the big blob
    assert not mask[5, 5]  # background corner
    assert 0.0 < mask.mean() < 0.9  # neither empty nor everything


def test_force_no_mask_returns_all_true():
    thumb = _synthetic_thumbnail()
    mask = build_tissue_mask(thumb, MaskingConfig(force_no_mask=True))
    assert mask.all()
    assert mask.shape == thumb.shape[:2]


def test_filter_grays_toggle_changes_mask():
    thumb = _synthetic_thumbnail()
    default_mask = build_tissue_mask(thumb, MaskingConfig(filter_grays=True))
    nogray_mask = build_tissue_mask(thumb, MaskingConfig(filter_grays=False))
    # Disabling the gray filter can only keep *more* (or equal) tissue.
    assert nogray_mask.sum() >= default_mask.sum()


def test_dilation_disabled_when_kernel_le_1():
    thumb = _synthetic_thumbnail()
    dilated = build_tissue_mask(thumb, MaskingConfig(dilation_kernel_size=5))
    undilated = build_tissue_mask(thumb, MaskingConfig(dilation_kernel_size=1))
    assert dilated.sum() >= undilated.sum()


def test_filter_grays_primitive():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 0] = (100, 100, 100)   # gray  -> filtered out (False)
    img[0, 1] = (200, 50, 180)    # chromatic -> kept (True)
    keep = filter_grays(img, tolerance=15)
    assert not keep[0, 0]
    assert keep[0, 1]


def test_otsu_mask_requires_uint8():
    with pytest.raises(TypeError):
        otsu_mask(np.zeros((4, 4, 3), dtype=np.float32))


def test_visualize_mask_blacks_out_background():
    thumb = _synthetic_thumbnail()
    mask = np.zeros(thumb.shape[:2], dtype=bool)
    mask[40:60, 40:60] = True
    vis = visualize_mask(thumb, mask)
    assert vis.shape == thumb.shape
    assert (vis[~mask] == 0).all()          # background blacked out
    assert np.array_equal(vis[mask], thumb[mask])  # tissue preserved


def test_visualize_mask_shape_mismatch_raises():
    thumb = _synthetic_thumbnail()
    with pytest.raises(ValueError):
        visualize_mask(thumb, np.zeros((10, 10), dtype=bool))


# --- skimage-version-agnostic morphology semantics (min_size -> max_size) ---
# These assert the threshold semantics directly, independent of which skimage
# version supplies the primitive, so the 0.26 `max_size` translation is verified.

def test_remove_small_objects_keeps_size_ge_min_size():
    mask = np.zeros((10, 40), dtype=bool)
    mask[1, 1:4] = True     # blob of 3 px
    mask[4, 10:14] = True   # blob of 4 px
    mask[7, 20:25] = True   # blob of 5 px
    out = remove_small_objects(mask, min_size=4, avoid_overmask=False)
    sizes = sorted(np.bincount(label(out).ravel())[1:].tolist())
    assert sizes == [4, 5]  # size-3 blob removed; sizes >= 4 kept


def test_fill_small_holes_fills_size_lt_min_size():
    mask = np.ones((12, 12), dtype=bool)
    mask[2, 2:5] = False    # interior hole of 3 px
    mask[6, 2:7] = False    # interior hole of 5 px
    out = fill_small_holes(mask, min_size=4)
    # hole of 3 (< 4) filled; hole of 5 kept -> 5 background pixels remain
    assert int((~out).sum()) == 5


def test_build_tissue_mask_emits_no_deprecation_warning():
    thumb = _synthetic_thumbnail()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        # filter_grays=False keeps more tissue so the morphology steps actually run
        build_tissue_mask(thumb, MaskingConfig(filter_grays=False))
