"""Command-line interface: estimate tissue area (mm²) for a slide or a folder.

The input can be a single slide or a folder (searched recursively). Results go to
one output directory:

    area.csv          per-slide areas (headline + total + largest + all regions)
    thumbnails/       labelled thumbnails (unless --skip-png)
    run_config.txt    the resolved run configuration (also printed at start)
    failures.csv      any slides that failed (only if some did)
    area.json         full per-slide records (only with --json)
"""

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__
from .area import tissue_area_for_slide
from .config import TISSUE_TYPES, masking_config_for_type

try:  # progress bar is optional; falls back to plain [i/N] lines
    from tqdm import tqdm as _tqdm
except Exception:  # pragma: no cover - tqdm is a declared dep but stay graceful
    _tqdm = None

# WSI extensions OpenSlide can read (used when --input is a folder).
_SLIDE_EXTS = (
    ".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu",
    ".scn", ".mrxs", ".svslide", ".bif",
)

_DEFAULT_OUTPUT = "tissuearea_output"

_TYPE_LABEL = {"ff": "fresh-frozen", "ffpe": "FFPE (paraffin)"}

# Per-slide area.csv columns. ``tissue_area_mm2`` is the headline value (mirrors
# --mode) so a non-expert has one obvious number; ``path`` traces each row back to
# its source file. ``whole_mm2`` is total tissue, ``largest_cc_mm2`` the largest
# region, ``section_areas_mm2`` every region (mm², largest-first, ';'-separated).
_CSV_FIELDS = [
    "slide_id",
    "path",
    "tissue_area_mm2",
    "n_sections",
    "whole_mm2",
    "largest_cc_mm2",
    "top2_sum_mm2",
    "section_areas_mm2",
    "mask_fraction",
    "width",
    "height",
    "mpp_x",
    "mpp_y",
    "mask_scale",
]

_FAIL_FIELDS = ["path", "slide_id", "error"]

_AREA_MODES = {
    "whole": "whole_mm2",
    "largest_cc": "largest_cc_mm2",
    "top2": "top2_sum_mm2",
}

_EPILOG = """\
examples:
  tissuearea slides/                         # a folder (searched recursively)
  tissuearea -i slide.svs -o results/        # a single slide
  tissuearea -i slides/ -o results/ -t ffpe  # FFPE cohort (gray filter on)
  tissuearea -i slides/ -o results/ --jobs 8 # process 8 slides in parallel
  tissuearea -i slides/ -o results/ --resume # continue an interrupted run
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tissuearea",
        description="Estimate physical tissue area (mm²) from whole-slide images.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Input can be positional (tissuearea slides/) or -i (tissuearea -i slides/).
    p.add_argument("input_pos", nargs="?", default=None, metavar="INPUT",
                   help="A slide file OR a folder of slides (same as -i).")
    p.add_argument("-i", "--input", dest="input_flag", default=None, metavar="INPUT",
                   help="A single slide file OR a folder of slides.")
    p.add_argument(
        "-o", "--output", default=_DEFAULT_OUTPUT,
        help=f"Output directory (default: ./{_DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "-t", "--type", dest="tissue_type",
        choices=list(TISSUE_TYPES), default="ff",
        help="Tissue preparation: 'ff' = fresh-frozen (default; gray filter off) "
        "or 'ffpe' = FFPE/paraffin (gray filter on). Picks the optimal segmentation.",
    )
    p.add_argument(
        "--no-recursive", action="store_true",
        help="When -i is a folder, do NOT search subfolders (default: recursive).",
    )
    p.add_argument(
        "--jobs", "-j", type=int, default=1, metavar="N",
        help="Process N slides in parallel (default: 1).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip slides already present in an existing area.csv (continue a run).",
    )
    p.add_argument(
        "--skip-png", action="store_true",
        help="Do NOT save the labelled thumbnail PNGs (they are saved by default).",
    )
    p.add_argument(
        "--mpp", type=float, default=None, metavar="MPP",
        help="Fallback microns-per-pixel for slides that lack MPP/objective metadata.",
    )
    p.add_argument(
        "--mode", choices=sorted(_AREA_MODES), default="largest_cc",
        help="Which area is the headline 'tissue_area_mm2' value (default: largest_cc).",
    )
    p.add_argument(
        "--scale", type=int, default=None,
        help="Override the thumbnail downsampling factor (default: 32).",
    )
    p.add_argument(
        "--label-min-area", type=float, default=0.0, metavar="MM2",
        help="In thumbnails, only label regions >= this many mm² (default 0 = all).",
    )
    p.add_argument("--json", action="store_true",
                   help="Also write area.json (full per-slide records).")
    p.add_argument("--quiet", action="store_true",
                   help="Minimal output (no banner, no progress bar).")
    p.add_argument("--version", action="version", version=f"tissuearea {__version__}")
    return p


def _resolve_inputs(input_path: str, recursive: bool = True) -> Tuple[List[str], str]:
    """Return ``(sorted slide paths, kind)`` for a file or folder.

    Folders are searched recursively by default. Raises ``FileNotFoundError``
    (with an actionable message) if the path is missing or has no slides.
    """
    p = Path(input_path)
    if p.is_dir():
        walker = p.rglob("*") if recursive else p.iterdir()
        slides = sorted(
            str(f) for f in walker
            if f.is_file() and f.suffix.lower() in _SLIDE_EXTS
        )
        if slides:
            return slides, "folder"
        exts = ", ".join(_SLIDE_EXTS)
        if not recursive and any(c.is_dir() for c in p.iterdir()):
            raise FileNotFoundError(
                f"No slides ({exts}) at the top level of {p}. It has subfolders — "
                f"drop --no-recursive to search them."
            )
        raise FileNotFoundError(f"No slides ({exts}) found in folder: {p}")
    if p.is_file():
        return [str(p)], "file"
    raise FileNotFoundError(f"Input not found: {p}")


def _thumb_name(path: str, used: set) -> str:
    """Collision-safe ``{stem}_regions.png`` (disambiguated by a path hash)."""
    stem = Path(path).stem
    name = f"{stem}_regions.png"
    if name in used:
        h = hashlib.md5(os.path.abspath(path).encode()).hexdigest()[:6]
        name = f"{stem}_{h}_regions.png"
    used.add(name)
    return name


def _config_report(args, slides, config, save_png, recursive, mpp, max_list=None) -> str:
    out_abs = os.path.abspath(args.output)
    lines = [
        "tissuearea — run configuration",
        "=" * 48,
        f"version         : {__version__}",
        f"timestamp       : {_dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "[input / output]",
        f"input           : {args.input_flag or args.input_pos}",
        f"resolved slides : {len(slides)}"
        + ("" if len(slides) != 1 else "  (single slide)")
        + (f"  (recursive)" if recursive else "  (top level only)"),
        f"output dir      : {out_abs}",
        f"thumbnails      : {os.path.join(out_abs, 'thumbnails') if save_png else '(skipped: --skip-png)'}",
        f"parallel jobs   : {args.jobs}",
        f"resume          : {args.resume}",
        "",
        "[segmentation]",
        f"tissue type     : {args.tissue_type}  ({_TYPE_LABEL[args.tissue_type]})",
        f"gray filter     : {'ON' if config.filter_grays else 'OFF'}"
        f"   <- assumed from --type; change with -t ff|ffpe",
        f"mask_scale      : {config.mask_scale}",
        f"mpp fallback    : {mpp if mpp else '(none; read from each slide)'}",
        "",
        "[area]",
        f"headline (mode) : tissue_area_mm2 = {args.mode}",
        f"label_min_area  : {args.label_min_area} mm2",
        "",
        "slides:",
    ]
    names = [Path(s).name for s in slides]
    if max_list is not None and len(names) > max_list:
        shown = names[:max_list]
        lines += [f"  - {n}" for n in shown]
        lines.append(f"  ... and {len(names) - max_list} more (full list in run_config.txt)")
    else:
        lines += [f"  - {n}" for n in names]
    return "\n".join(lines)


def _process_slide(path, config, labeled_path, label_min_area, mpp_fallback, include_regions=False):
    """Worker (module-level so it is picklable for --jobs)."""
    out = tissue_area_for_slide(
        path,
        config=config,
        labeled_output_path=labeled_path,
        label_min_area_mm2=label_min_area,
        mpp_fallback=mpp_fallback,
        include_regions=include_regions,
    )
    out["path"] = os.path.abspath(path)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_path = args.input_flag or args.input_pos
    if not input_path:
        parser.print_help()
        return 0

    recursive = not args.no_recursive
    try:
        slides, _kind = _resolve_inputs(input_path, recursive=recursive)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    overrides = {}
    if args.scale is not None:
        overrides["mask_scale"] = args.scale
    config = masking_config_for_type(args.tissue_type, **overrides)

    save_png = not args.skip_png
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    thumb_dir = os.path.join(out_dir, "thumbnails")
    if save_png:
        os.makedirs(thumb_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "area.csv")

    # --resume: skip slides already recorded in an existing area.csv.
    done_paths = set()
    if args.resume and os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("path"):
                    done_paths.add(os.path.abspath(row["path"]))

    used_names: set = set()
    todo: List[Tuple[str, Optional[str]]] = []
    skipped = 0
    for p in slides:
        if os.path.abspath(p) in done_paths:
            skipped += 1
            continue
        tp = os.path.join(thumb_dir, _thumb_name(p, used_names)) if save_png else None
        todo.append((p, tp))

    # Config banner: print (capped) + persist full copy to run_config.txt.
    report_full = _config_report(args, slides, config, save_png, recursive, args.mpp)
    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write(report_full + "\n")
    if not args.quiet:
        print(_config_report(args, slides, config, save_png, recursive, args.mpp, max_list=12))
        if skipped:
            print(f"resume: skipping {skipped} slide(s) already in area.csv")
        print("-" * 48)

    headline_key = _AREA_MODES[args.mode]

    # Stream area.csv (append when resuming an existing file).
    append_csv = args.resume and os.path.exists(csv_path)
    f_csv = open(csv_path, "a" if append_csv else "w", newline="")
    writer = csv.DictWriter(f_csv, fieldnames=_CSV_FIELDS, extrasaction="ignore")
    if not append_csv:
        writer.writeheader()

    results = []
    failures = []
    fail_state = {"writer": None, "f": None}
    bar = _tqdm(total=len(todo), desc="tissue-area", unit="slide") if (_tqdm and not args.quiet) else None

    def _write_row(out):
        out["tissue_area_mm2"] = out[headline_key]
        row = {k: out.get(k) for k in _CSV_FIELDS}
        row["section_areas_mm2"] = ";".join(f"{a:.6g}" for a in out.get("section_areas_mm2", []))
        writer.writerow(row)
        f_csv.flush()
        results.append(out)

    def _write_failure(path, err):
        if fail_state["writer"] is None:
            fail_path = os.path.join(out_dir, "failures.csv")
            append_fail = args.resume and os.path.exists(fail_path)
            fh = open(fail_path, "a" if append_fail else "w", newline="")
            fw = csv.DictWriter(fh, fieldnames=_FAIL_FIELDS)
            if not append_fail:
                fw.writeheader()
            fail_state["writer"], fail_state["f"] = fw, fh
        fail_state["writer"].writerow(
            {"path": os.path.abspath(path), "slide_id": Path(path).stem, "error": err}
        )
        fail_state["f"].flush()
        failures.append((path, err))

    def _report(path, out=None, err=None):
        if err is not None:
            msg = f"ERROR  {Path(path).name}: {err}"
            (bar.write(msg) if bar else print(msg, file=sys.stderr))
            _write_failure(path, err)
        else:
            _write_row(out)
            if bar:
                bar.set_postfix_str(f"{out['slide_id']}: {out['tissue_area_mm2']:.1f} mm2")
            elif not args.quiet:
                n = len(results) + len(failures)
                print(f"[{n}/{len(todo)}] {out['slide_id']}\t{args.mode}={out['tissue_area_mm2']:.2f} mm²")
        if bar:
            bar.update(1)

    if args.jobs and args.jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = {
                ex.submit(_process_slide, p, config, tp, args.label_min_area, args.mpp, args.json): p
                for (p, tp) in todo
            }
            for fut in as_completed(futs):
                p = futs[fut]
                try:
                    _report(p, out=fut.result())
                except Exception as e:  # noqa: BLE001
                    _report(p, err=str(e))
    else:
        for (p, tp) in todo:
            try:
                _report(p, out=_process_slide(p, config, tp, args.label_min_area, args.mpp, args.json))
            except Exception as e:  # noqa: BLE001
                _report(p, err=str(e))

    if bar:
        bar.close()
    f_csv.close()
    if fail_state["f"]:
        fail_state["f"].close()

    if args.json:
        with open(os.path.join(out_dir, "area.json"), "w") as f:
            json.dump(results, f, indent=2)

    # Final summary.
    print("-" * 48)
    print(f"Done: {len(results)} ok, {len(failures)} failed"
          + (f", {skipped} skipped (resume)" if skipped else "")
          + f"  ->  {os.path.abspath(csv_path)}")
    if save_png and results:
        print(f"Thumbnails -> {os.path.abspath(thumb_dir)}")
    if failures:
        print(f"Failures listed in {os.path.join(os.path.abspath(out_dir), 'failures.csv')}",
              file=sys.stderr)

    # Honest exit code: nonzero if anything failed (or nothing could be produced).
    if failures:
        return 1 if results else 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
