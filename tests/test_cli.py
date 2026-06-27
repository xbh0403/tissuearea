"""Smoke tests for the CLI argument handling (no real slide needed)."""

import pytest

from tissuearea.cli import _build_parser, main


def test_parser_defaults():
    args = _build_parser().parse_args(["a.svs", "b.svs"])
    assert args.slides == ["a.svs", "b.svs"]
    assert args.mode == "largest_cc"
    assert args.no_grays is False
    assert args.scale is None


def test_parser_flags():
    args = _build_parser().parse_args(["s.svs", "--no-grays", "--scale", "16", "--mode", "whole"])
    assert args.no_grays is True
    assert args.scale == 16
    assert args.mode == "whole"


def test_parser_labeling_flags():
    args = _build_parser().parse_args(
        ["s.svs", "--save-labeled", "out/labels", "--label-min-area", "0.5"]
    )
    assert args.save_labeled == "out/labels"
    assert args.label_min_area == 0.5


def test_csv_has_total_largest_and_all_regions():
    # The CSV column set must cover total, largest, and every region's area.
    from tissuearea.cli import _CSV_FIELDS

    assert "whole_mm2" in _CSV_FIELDS          # total tissue area
    assert "largest_cc_mm2" in _CSV_FIELDS     # largest piece
    assert "section_areas_mm2" in _CSV_FIELDS  # all regions


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "tissuearea" in capsys.readouterr().out


def test_missing_slide_reports_failure(tmp_path, capsys):
    # A nonexistent slide should be reported and yield a nonzero exit, not crash.
    rc = main([str(tmp_path / "does_not_exist.svs")])
    assert rc == 1
    assert "ERROR" in capsys.readouterr().err
