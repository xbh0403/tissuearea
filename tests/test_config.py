"""Tests for tissue-type -> MaskingConfig preset selection."""

import pytest

from tissuearea import MaskingConfig, masking_config_for_type
from tissuearea.config import TISSUE_TYPES


def test_ff_turns_gray_filter_off():
    cfg = masking_config_for_type("ff")
    assert isinstance(cfg, MaskingConfig)
    assert cfg.filter_grays is False


def test_ffpe_keeps_gray_filter_on():
    assert masking_config_for_type("ffpe").filter_grays is True


def test_default_is_fresh_frozen():
    # "make no-grays the default" -> the default preset is fresh-frozen.
    assert masking_config_for_type().filter_grays is False


@pytest.mark.parametrize("alias,expected", [
    ("FF", False), ("Frozen", False), ("fresh-frozen", False),
    ("FFPE", True), ("paraffin", True),
])
def test_aliases(alias, expected):
    assert masking_config_for_type(alias).filter_grays is expected


def test_overrides_apply_on_top_of_preset():
    cfg = masking_config_for_type("ff", mask_scale=16, dilation_kernel_size=1)
    assert cfg.filter_grays is False       # preset preserved
    assert cfg.mask_scale == 16            # override applied
    assert cfg.dilation_kernel_size == 1


def test_unknown_type_raises():
    with pytest.raises(ValueError):
        masking_config_for_type("bogus")


def test_defaults_unchanged_are_production():
    # The bare MaskingConfig must still match production (gray filter ON), so
    # build_tissue_mask keeps reproducing the source pipeline byte-for-byte.
    assert MaskingConfig().filter_grays is True
    assert "ff" in TISSUE_TYPES and "ffpe" in TISSUE_TYPES
