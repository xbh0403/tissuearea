"""Tests for the physical-area math (deterministic, no real slide needed)."""

import numpy as np
import pytest

from tissuearea import area_from_mask, resolve_mpp_xy, tissue_area_from_thumbnail
from tissuearea.config import MaskingConfig
from tissuearea.masking import build_tissue_mask

# A scale-32 thumbnail of shape (mh=50, mw=100) over a level-0 slide of
# (W=3200, H=1600) at 0.25 µm/px gives per-mask-pixel area of:
#   (3200*0.25/100) * (1600*0.25/50) = 8 * 8 = 64 µm²
MW, MH = 100, 50
W, H = 3200, 1600
MPP = 0.25
PER_PX_UM2 = (W * MPP / MW) * (H * MPP / MH)  # 64.0


def test_per_pixel_constant_sanity():
    assert PER_PX_UM2 == pytest.approx(64.0)


def test_area_from_mask_full_tissue():
    mask = np.ones((MH, MW), dtype=bool)
    out = area_from_mask(mask, W, H, MPP, MPP)
    # whole slide is physically (3200*0.25) x (1600*0.25) µm = 800 x 400 µm
    assert out["whole_mm2"] == pytest.approx(800 * 400 / 1e6)  # 0.32
    assert out["whole_mm2"] == pytest.approx(MW * MH * PER_PX_UM2 / 1e6)
    assert out["mask_fraction"] == pytest.approx(1.0)
    assert out["n_sections"] == 1
    assert out["largest_cc_mm2"] == pytest.approx(out["whole_mm2"])
    assert out["mask_w"] == MW and out["mask_h"] == MH


def test_area_from_mask_single_block():
    mask = np.zeros((MH, MW), dtype=bool)
    mask[10:20, 5:25] = True  # 10 rows x 20 cols = 200 px
    out = area_from_mask(mask, W, H, MPP, MPP)
    assert out["whole_mm2"] == pytest.approx(200 * PER_PX_UM2 / 1e6)  # 0.0128
    assert out["n_sections"] == 1
    assert out["largest_cc_mm2"] == pytest.approx(out["whole_mm2"])
    assert out["top2_sum_mm2"] == pytest.approx(out["whole_mm2"])


def test_area_from_mask_two_sections_largest_and_top2():
    mask = np.zeros((MH, MW), dtype=bool)
    mask[5:15, 5:15] = True    # block A: 10x10 = 100 px
    mask[30:35, 50:55] = True  # block B:  5x5 =  25 px (separated)
    out = area_from_mask(mask, W, H, MPP, MPP)
    a = 100 * PER_PX_UM2 / 1e6
    b = 25 * PER_PX_UM2 / 1e6
    assert out["n_sections"] == 2
    assert out["whole_mm2"] == pytest.approx(a + b)
    assert out["largest_cc_mm2"] == pytest.approx(a)
    assert out["top2_sum_mm2"] == pytest.approx(a + b)
    assert out["section_areas_mm2"][0] == pytest.approx(a)
    assert out["section_areas_mm2"][1] == pytest.approx(b)


def test_area_from_mask_empty():
    mask = np.zeros((MH, MW), dtype=bool)
    out = area_from_mask(mask, W, H, MPP, MPP)
    assert out["whole_mm2"] == 0.0
    assert out["largest_cc_mm2"] == 0.0
    assert out["top2_sum_mm2"] == 0.0
    assert out["n_sections"] == 0
    assert out["section_areas_mm2"] == []


def test_area_from_mask_anisotropic_mpp():
    mask = np.ones((MH, MW), dtype=bool)
    out = area_from_mask(mask, W, H, mpp_x=0.25, mpp_y=0.50)
    per_px = (W * 0.25 / MW) * (H * 0.50 / MH)  # 8 * 16 = 128
    assert out["whole_mm2"] == pytest.approx(MW * MH * per_px / 1e6)  # 0.64


def test_area_from_mask_non_2d_raises():
    with pytest.raises(ValueError):
        area_from_mask(np.ones((4, 4, 3), dtype=bool), W, H, MPP, MPP)


def test_resolve_mpp_both_present():
    props = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.30"}
    assert resolve_mpp_xy(props) == pytest.approx((0.25, 0.30))


def test_resolve_mpp_one_missing_mirrors_other():
    props = {"openslide.mpp-x": "0.262766"}
    mx, my = resolve_mpp_xy(props)
    assert mx == pytest.approx(0.262766)
    assert my == pytest.approx(0.262766)


def test_resolve_mpp_fallback_to_objective_power():
    props = {"openslide.objective-power": "40"}
    mx, my = resolve_mpp_xy(props)
    assert mx == pytest.approx(0.25)  # 10 / 40
    assert my == pytest.approx(0.25)


def test_resolve_mpp_explicit_fallback():
    mx, my = resolve_mpp_xy({}, fallback_mpp=0.5)
    assert (mx, my) == pytest.approx((0.5, 0.5))


def test_resolve_mpp_none_raises():
    with pytest.raises(ValueError):
        resolve_mpp_xy({})


def test_include_regions_matches_section_areas():
    out = tissue_area_from_thumbnail(_synthetic_thumbnail(), W, H, MPP, MPP, include_regions=True)
    assert "regions" in out
    # the per-region areas mirror the flat section list (and the thumbnail #labels)
    assert [r["area_mm2"] for r in out["regions"]] == pytest.approx(out["section_areas_mm2"])
    assert [r["rank"] for r in out["regions"]] == list(range(1, len(out["regions"]) + 1))


def test_regions_absent_by_default():
    out = tissue_area_from_thumbnail(_synthetic_thumbnail(), W, H, MPP, MPP)
    assert "regions" not in out


def test_tissue_area_from_thumbnail_composes_mask_and_area():
    thumb = _synthetic_thumbnail()
    cfg = MaskingConfig()
    mask = build_tissue_mask(thumb, config=cfg)
    expected = area_from_mask(mask, W, H, MPP, MPP)
    out = tissue_area_from_thumbnail(thumb, W, H, MPP, MPP, config=cfg)
    assert out["whole_mm2"] == pytest.approx(expected["whole_mm2"])
    assert out["largest_cc_mm2"] == pytest.approx(expected["largest_cc_mm2"])
    assert out["n_sections"] == expected["n_sections"]


def _synthetic_thumbnail(seed: int = 0) -> np.ndarray:
    """A small RGB thumbnail with a pinkish tissue blob on a near-white field."""
    rng = np.random.default_rng(seed)
    thumb = np.full((120, 200, 3), 235, dtype=np.uint8)  # near-white background
    thumb[30:90, 20:110] = np.array([180, 110, 170], dtype=np.uint8)   # H&E-ish blob
    thumb[60:80, 150:180] = np.array([170, 100, 160], dtype=np.uint8)  # smaller blob
    noise = rng.integers(-12, 12, size=thumb.shape, dtype=np.int16)
    return np.clip(thumb.astype(np.int16) + noise, 0, 255).astype(np.uint8)
