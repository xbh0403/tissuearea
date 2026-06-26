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
