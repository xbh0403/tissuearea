"""Smoke tests for the CLI argument handling (no real slide needed)."""

import os

import pytest

from tissuearea.cli import _DEFAULT_OUTPUT, _CSV_FIELDS, _build_parser, _resolve_inputs, _thumb_name, main


def test_parser_defaults():
    args = _build_parser().parse_args(["slide.svs"])          # positional input
    assert args.input_pos == "slide.svs"
    assert args.output == _DEFAULT_OUTPUT
    assert args.tissue_type == "ff"       # fresh-frozen (gray filter off) is default
    assert args.skip_png is False         # thumbnails saved by default
    assert args.no_recursive is False     # recursive by default
    assert args.no_json is False          # area.json written by default
    assert args.resume is False
    assert args.jobs == 8
    assert args.mode == "largest_cc"


def test_parser_input_flag_and_flags():
    args = _build_parser().parse_args(
        ["-i", "d/", "-o", "out", "-t", "ffpe", "--skip-png", "--no-recursive",
         "--jobs", "4", "--resume", "--mpp", "0.5", "--mode", "whole", "--no-json"]
    )
    assert args.input_flag == "d/"
    assert args.output == "out"
    assert args.tissue_type == "ffpe"
    assert args.skip_png is True
    assert args.no_recursive is True
    assert args.jobs == 4
    assert args.resume is True
    assert args.mpp == 0.5
    assert args.mode == "whole"
    assert args.no_json is True


def test_parser_rejects_unknown_type():
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["s.svs", "-t", "bogus"])


def test_no_args_prints_help_and_exits_zero(capsys):
    rc = main([])
    assert rc == 0
    out = capsys.readouterr().out
    assert "usage" in out.lower() and "examples" in out.lower()


def test_csv_has_headline_path_total_largest_and_all_regions():
    for col in ("tissue_area_mm2", "path", "whole_mm2", "largest_cc_mm2", "section_areas_mm2"):
        assert col in _CSV_FIELDS
    # headline + path lead the table for easy reading/joining
    assert _CSV_FIELDS[0] == "slide_id"
    assert _CSV_FIELDS[1] == "path"
    assert _CSV_FIELDS[2] == "tissue_area_mm2"


def test_resolve_inputs_single_file(tmp_path):
    f = tmp_path / "a.svs"
    f.write_bytes(b"")
    slides, kind = _resolve_inputs(str(f))
    assert slides == [str(f)] and kind == "file"


def test_resolve_inputs_folder_is_recursive(tmp_path):
    (tmp_path / "top.svs").write_bytes(b"")
    (tmp_path / "caseA").mkdir()
    (tmp_path / "caseA" / "nested.ndpi").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")  # ignored (not a slide ext)
    slides, kind = _resolve_inputs(str(tmp_path), recursive=True)
    assert kind == "folder"
    names = sorted(os.path.basename(s) for s in slides)  # native sep: win \, posix /
    assert names == ["nested.ndpi", "top.svs"]   # subfolder slide found


def test_resolve_inputs_non_recursive_hints_at_subfolders(tmp_path):
    (tmp_path / "caseA").mkdir()
    (tmp_path / "caseA" / "s.svs").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="subfolders"):
        _resolve_inputs(str(tmp_path), recursive=False)


def test_resolve_inputs_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_inputs(str(tmp_path / "nope.svs"))


def test_thumb_name_is_collision_safe():
    used = set()
    a = _thumb_name("/x/caseA/slide.svs", used)
    b = _thumb_name("/y/caseB/slide.svs", used)  # same stem, different path
    assert a == "slide_regions.png"
    assert b != a and b.endswith("_regions.png")


def test_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "tissuearea" in capsys.readouterr().out


def test_missing_input_reports_error(tmp_path, capsys):
    rc = main(["-i", str(tmp_path / "does_not_exist.svs"), "-o", str(tmp_path / "out")])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err
