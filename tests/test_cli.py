"""Smoke tests for the CLI argument handling (no real slide needed)."""

import pytest

from tissuearea.cli import _DEFAULT_OUTPUT, _CSV_FIELDS, _build_parser, _resolve_inputs, main


def test_parser_defaults():
    args = _build_parser().parse_args(["-i", "slide.svs"])
    assert args.input == "slide.svs"
    assert args.output == _DEFAULT_OUTPUT
    assert args.tissue_type == "ff"       # fresh-frozen (gray filter off) is default
    assert args.skip_png is False         # thumbnails saved by default
    assert args.mode == "largest_cc"
    assert args.scale is None


def test_parser_flags():
    args = _build_parser().parse_args(
        ["-i", "d/", "-o", "out", "-t", "ffpe", "--skip-png", "--scale", "16", "--mode", "whole"]
    )
    assert args.input == "d/"
    assert args.output == "out"
    assert args.tissue_type == "ffpe"
    assert args.skip_png is True
    assert args.scale == 16
    assert args.mode == "whole"


def test_input_is_required():
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])


def test_parser_rejects_unknown_type():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["-i", "s.svs", "-t", "bogus"])


def test_csv_has_total_largest_and_all_regions():
    # The CSV column set must cover total, largest, and every region's area.
    assert "whole_mm2" in _CSV_FIELDS          # total tissue area
    assert "largest_cc_mm2" in _CSV_FIELDS     # largest piece
    assert "section_areas_mm2" in _CSV_FIELDS  # all regions


def test_resolve_inputs_single_file(tmp_path):
    f = tmp_path / "a.svs"
    f.write_bytes(b"")  # existence is all _resolve_inputs checks
    slides, kind = _resolve_inputs(str(f))
    assert slides == [str(f)] and kind == "file"


def test_resolve_inputs_folder(tmp_path):
    (tmp_path / "a.svs").write_bytes(b"")
    (tmp_path / "b.ndpi").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")  # ignored (not a slide ext)
    slides, kind = _resolve_inputs(str(tmp_path))
    assert kind == "folder"
    assert [s.rsplit("/", 1)[-1] for s in slides] == ["a.svs", "b.ndpi"]


def test_resolve_inputs_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_inputs(str(tmp_path))


def test_resolve_inputs_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_inputs(str(tmp_path / "nope.svs"))


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "tissuearea" in capsys.readouterr().out


def test_missing_input_reports_error(tmp_path, capsys):
    rc = main(["-i", str(tmp_path / "does_not_exist.svs"), "-o", str(tmp_path / "out")])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err
