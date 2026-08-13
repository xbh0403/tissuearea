"""Smoke tests for the CLI argument handling (no real slide needed)."""

import csv
import json
import os
from pathlib import Path

import pytest

from tissuearea import cli
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


# --- batch/resume behaviour -------------------------------------------------
# These drive the real main(): real work planning, CSV streaming, JSON writing
# and exit codes. Only the one step that needs a real WSI on disk — the
# per-slide worker — is replaced, so nothing under test is faked out.

def _fake_record(path, area_mm2=12.5):
    """A complete per-slide record, exactly as ``_process_slide`` returns one."""
    return {
        "slide_id": Path(path).stem,
        "path": os.path.abspath(path),
        "whole_mm2": area_mm2,
        "largest_cc_mm2": area_mm2,
        "top2_sum_mm2": area_mm2,
        "n_sections": 1,
        "section_areas_mm2": [area_mm2],
        "mask_fraction": 0.25,
        "mask_w": 100,
        "mask_h": 50,
        "width": 3200,
        "height": 1600,
        "mpp_x": 0.25,
        "mpp_y": 0.25,
        "mask_scale": 32,
        "regions": [
            {"rank": 1, "label": 1, "n_pixels": 100, "area_mm2": area_mm2,
             "centroid_xy": (10.0, 10.0), "bbox": (5, 5, 15, 15)},
        ],
    }


def _fake_worker(monkeypatch, failing=()):
    """Swap in a worker that never opens a slide; returns {abs_path: thumb_path}."""
    seen = {}

    def fake(path, config, labeled_path, label_min_area, mpp_fallback, include_regions=False):
        ap = os.path.abspath(path)
        seen[ap] = labeled_path
        if ap in failing:
            raise RuntimeError("unreadable slide")
        return _fake_record(path)

    monkeypatch.setattr(cli, "_process_slide", fake)
    return seen


def _paths_in_json(out_dir):
    with open(os.path.join(str(out_dir), "area.json")) as f:
        return sorted(r["path"] for r in json.load(f))


def _paths_in_csv(out_dir):
    with open(os.path.join(str(out_dir), "area.csv"), newline="") as f:
        return sorted(row["path"] for row in csv.DictReader(f))


def test_resume_keeps_records_written_before_the_interruption(tmp_path, monkeypatch):
    # area.csv is appended on resume; area.json must not be truncated to the
    # slides this run happened to process.
    slides = tmp_path / "slides"
    slides.mkdir()
    a, b = slides / "a.svs", slides / "b.svs"
    a.write_bytes(b"")
    b.write_bytes(b"")
    out = tmp_path / "out"
    a_abs, b_abs = os.path.abspath(str(a)), os.path.abspath(str(b))

    _fake_worker(monkeypatch, failing={b_abs})
    rc = main(["-i", str(slides), "-o", str(out), "--jobs", "1", "--skip-png", "--quiet"])
    assert rc == 1                          # 'b' failed, 'a' succeeded
    assert _paths_in_json(out) == [a_abs]

    _fake_worker(monkeypatch)               # 'b' is readable on the second attempt
    rc = main(["-i", str(slides), "-o", str(out), "--jobs", "1", "--skip-png",
               "--quiet", "--resume"])
    assert rc == 0
    assert _paths_in_csv(out) == [a_abs, b_abs]
    assert _paths_in_json(out) == [a_abs, b_abs]


def test_resume_refreshes_a_reprocessed_record_instead_of_duplicating_it(tmp_path, monkeypatch):
    slides = tmp_path / "slides"
    slides.mkdir()
    a = slides / "a.svs"
    a.write_bytes(b"")
    out = tmp_path / "out"
    a_abs = os.path.abspath(str(a))

    _fake_worker(monkeypatch)
    main(["-i", str(slides), "-o", str(out), "--jobs", "1", "--skip-png", "--quiet"])

    # Same slide processed again under --resume (its area.csv row was removed).
    os.remove(os.path.join(str(out), "area.csv"))
    monkeypatch.setattr(
        cli, "_process_slide",
        lambda path, *a_, **k_: _fake_record(path, area_mm2=99.0),
    )
    main(["-i", str(slides), "-o", str(out), "--jobs", "1", "--skip-png",
          "--quiet", "--resume"])

    with open(os.path.join(str(out), "area.json")) as f:
        records = json.load(f)
    assert [r["path"] for r in records] == [a_abs]      # one entry, not two
    assert records[0]["whole_mm2"] == 99.0              # the fresh value wins


def test_a_fresh_run_does_not_inherit_a_previous_runs_area_json(tmp_path, monkeypatch):
    out = tmp_path / "out"
    first = tmp_path / "first"
    first.mkdir()
    (first / "a.svs").write_bytes(b"")
    second = tmp_path / "second"
    second.mkdir()
    (second / "b.svs").write_bytes(b"")

    _fake_worker(monkeypatch)
    main(["-i", str(first), "-o", str(out), "--jobs", "1", "--skip-png", "--quiet"])
    # No --resume: this is a different cohort into the same output dir.
    main(["-i", str(second), "-o", str(out), "--jobs", "1", "--skip-png", "--quiet"])

    assert _paths_in_json(out) == [os.path.abspath(str(second / "b.svs"))]
    assert _paths_in_csv(out) == [os.path.abspath(str(second / "b.svs"))]


def test_resume_does_not_reuse_a_finished_slides_thumbnail_name(tmp_path, monkeypatch):
    # Two slides share a stem, so only one can own 'slide_regions.png'. If the
    # finished slide's name is not reserved, the resumed slide overwrites its PNG.
    root = tmp_path / "slides"
    (root / "caseA").mkdir(parents=True)
    (root / "caseB").mkdir(parents=True)
    a, b = root / "caseA" / "slide.svs", root / "caseB" / "slide.svs"
    a.write_bytes(b"")
    b.write_bytes(b"")
    a_abs, b_abs = os.path.abspath(str(a)), os.path.abspath(str(b))
    out = tmp_path / "out"

    seen = _fake_worker(monkeypatch, failing={b_abs})
    main(["-i", str(root), "-o", str(out), "--jobs", "1", "--quiet"])
    a_thumb = seen[a_abs]

    seen = _fake_worker(monkeypatch)
    main(["-i", str(root), "-o", str(out), "--jobs", "1", "--quiet", "--resume"])
    b_thumb = seen[b_abs]

    assert a_thumb is not None and b_thumb is not None
    assert b_thumb != a_thumb


def test_thumbnail_names_are_the_same_whether_or_not_a_run_was_resumed(tmp_path, monkeypatch):
    root = tmp_path / "slides"
    (root / "caseA").mkdir(parents=True)
    (root / "caseB").mkdir(parents=True)
    a, b = root / "caseA" / "slide.svs", root / "caseB" / "slide.svs"
    a.write_bytes(b"")
    b.write_bytes(b"")
    a_abs, b_abs = os.path.abspath(str(a)), os.path.abspath(str(b))

    # One uninterrupted run assigns the reference names.
    seen = _fake_worker(monkeypatch)
    main(["-i", str(root), "-o", str(tmp_path / "whole"), "--jobs", "1", "--quiet"])
    want = {p: Path(t).name for p, t in seen.items()}

    # An interrupted run + --resume must land on those same names.
    split = tmp_path / "split"
    seen = _fake_worker(monkeypatch, failing={b_abs})
    main(["-i", str(root), "-o", str(split), "--jobs", "1", "--quiet"])
    got = {a_abs: Path(seen[a_abs]).name}
    seen = _fake_worker(monkeypatch)
    main(["-i", str(root), "-o", str(split), "--jobs", "1", "--quiet", "--resume"])
    got[b_abs] = Path(seen[b_abs]).name

    assert got == want
