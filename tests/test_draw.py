"""Tests for region_areas and the labelled-thumbnail renderer."""

import numpy as np
import pytest
from PIL import Image

from tissuearea import area_from_mask, draw_region_labels, region_areas

# Same geometry constants as test_area.
MW, MH = 100, 50
W, H = 3200, 1600
MPP = 0.25


def _two_section_mask():
    mask = np.zeros((MH, MW), dtype=bool)
    mask[5:15, 5:15] = True    # block A: 10x10 = 100 px (larger)
    mask[30:35, 50:55] = True  # block B:  5x5 =  25 px
    return mask


def _rgb_thumb():
    return np.full((MH, MW, 3), 230, dtype=np.uint8)


def test_region_areas_match_area_from_mask_sections():
    mask = _two_section_mask()
    regs = region_areas(mask, W, H, MPP, MPP)
    sections = area_from_mask(mask, W, H, MPP, MPP)["section_areas_mm2"]
    assert [r["area_mm2"] for r in regs] == pytest.approx(sections)
    # ranked largest-first, ranks are 1..n
    assert [r["rank"] for r in regs] == [1, 2]
    assert regs[0]["area_mm2"] > regs[1]["area_mm2"]
    assert regs[0]["n_pixels"] == 100 and regs[1]["n_pixels"] == 25


def test_region_areas_empty():
    assert region_areas(np.zeros((MH, MW), dtype=bool), W, H, MPP, MPP) == []


def test_region_areas_centroid_inside_bbox():
    regs = region_areas(_two_section_mask(), W, H, MPP, MPP)
    for r in regs:
        cx, cy = r["centroid_xy"]
        min_row, min_col, max_row, max_col = r["bbox"]
        assert min_col <= cx <= max_col
        assert min_row <= cy <= max_row


def test_draw_region_labels_shape_and_no_mutation():
    thumb = _rgb_thumb()
    original = thumb.copy()
    mask = _two_section_mask()
    out = draw_region_labels(thumb, mask, W, H, MPP, MPP)
    assert out.shape == thumb.shape
    assert out.dtype == np.uint8
    # input thumbnail must be untouched
    assert np.array_equal(thumb, original)
    # something was drawn (contours / labels differ from the flat background)
    assert not np.array_equal(out, thumb)


def test_draw_region_labels_draws_contour_color():
    thumb = _rgb_thumb()
    mask = _two_section_mask()
    # Suppress text labels/header so the contours aren't overdrawn on this tiny
    # synthetic image; min_area huge => contours only.
    out = draw_region_labels(
        thumb, mask, W, H, MPP, MPP,
        contour_color=(255, 0, 0), min_area_mm2=1e9, show_header=False,
    )
    # at least one pure-red contour pixel exists
    red = (out[:, :, 0] == 255) & (out[:, :, 1] == 0) & (out[:, :, 2] == 0)
    assert red.any()


def test_draw_region_labels_empty_mask_runs():
    thumb = _rgb_thumb()
    out = draw_region_labels(thumb, np.zeros((MH, MW), dtype=bool), W, H, MPP, MPP)
    assert out.shape == thumb.shape


def test_draw_region_labels_shape_mismatch_raises():
    with pytest.raises(ValueError):
        draw_region_labels(_rgb_thumb(), np.zeros((10, 10), dtype=bool), W, H, MPP, MPP)


def test_draw_region_labels_writes_file(tmp_path):
    out_path = tmp_path / "regions.png"
    draw_region_labels(
        _rgb_thumb(), _two_section_mask(), W, H, MPP, MPP, output_path=str(out_path)
    )
    assert out_path.exists()
    with Image.open(out_path) as im:
        assert im.size == (MW, MH)  # PIL size is (width, height)
