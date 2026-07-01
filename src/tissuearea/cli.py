"""Command-line interface: estimate tissue area (mm²) for a slide or a folder.

Examples
--------
    tissuearea -i slide.svs
    tissuearea -i slides/ -o results/ -t ffpe
    tissuearea -i slides/ -o results/ --skip-png

Outputs (all under ``-o``):
    area.csv            per-slide areas (total, largest, all regions)
    thumbnails/         labelled thumbnails (unless --skip-png)
    run_config.txt      the resolved run configuration (also printed at start)
    area.json           full per-slide records (only with --json)
"""

import argparse
import csv
import datetime as _dt
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from . import __version__
from .area import tissue_area_for_slide
from .config import TISSUE_TYPES, masking_config_for_type

# WSI extensions OpenSlide can read (used when --input is a folder).
_SLIDE_EXTS = (
    ".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu",
    ".scn", ".mrxs", ".svslide", ".bif",
)

_DEFAULT_OUTPUT = "tissuearea_output"

# Per-slide CSV columns. ``whole_mm2`` is the total tissue area, ``largest_cc_mm2``
# the single largest region, and ``section_areas_mm2`` holds every region's area
# (mm², largest first) as a ';'-separated list.
_CSV_FIELDS = [
    "slide_id",
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

_AREA_MODES = {
    "whole": "whole_mm2",
    "largest_cc": "largest_cc_mm2",
    "top2": "top2_sum_mm2",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tissuearea",
        description="Estimate physical tissue area (mm²) from WSI tissue masks + MPP.",
    )
    p.add_argument(
        "-i", "--input", required=True,
        help="A single slide file OR a folder of slides.",
    )
    p.add_argument(
        "-o", "--output", default=_DEFAULT_OUTPUT,
        help=f"Output directory for area.csv, thumbnails/ and run_config.txt "
        f"(default: ./{_DEFAULT_OUTPUT}).",
    )
    p.add_argument(
        "-t", "--type", dest="tissue_type",
        choices=list(TISSUE_TYPES), default="ff",
        help="Tissue preparation type; selects the optimal segmentation. "
        "'ff' = fresh-frozen (default; gray filter OFF), 'ffpe' = FFPE (gray filter ON).",
    )
    p.add_argument(
        "--skip-png", action="store_true",
        help="Do NOT save the labelled thumbnail PNGs (they are saved by default).",
    )
    p.add_argument(
        "--mode", choices=sorted(_AREA_MODES), default="largest_cc",
        help="Which area candidate to print as the headline value (default: largest_cc).",
    )
    p.add_argument(
        "--scale", type=int, default=None,
        help="Override the thumbnail downsampling factor (default: MaskingConfig.mask_scale=32).",
    )
    p.add_argument(
        "--label-min-area", type=float, default=0.0, metavar="MM2",
        help="In thumbnails, only write area text for regions >= this many mm² "
        "(contours still drawn). Default 0 = label every region.",
    )
    p.add_argument(
        "--json", action="store_true",
        help="Also write area.json (full per-slide records) into the output dir.",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress the per-slide progress lines (config banner still shown).",
    )
    p.add_argument("--version", action="version", version=f"tissuearea {__version__}")
    return p


def _resolve_inputs(input_path: str) -> Tuple[List[str], str]:
    """Return ``(sorted slide paths, kind)`` for a file or folder.

    Raises:
        FileNotFoundError: if the path is missing, or a folder has no slides.
    """
    p = Path(input_path)
    if p.is_dir():
        slides = sorted(
            str(f) for f in p.iterdir()
            if f.is_file() and f.suffix.lower() in _SLIDE_EXTS
        )
        if not slides:
            raise FileNotFoundError(
                f"No slides ({', '.join(_SLIDE_EXTS)}) found in folder: {p}"
            )
        return slides, "folder"
    if p.is_file():
        return [str(p)], "file"
    raise FileNotFoundError(f"Input not found: {p}")


def _config_report(args, slides, config, save_png: bool) -> str:
    """Human-readable summary of the run, for stdout and run_config.txt."""
    out_abs = os.path.abspath(args.output)
    lines = [
        "tissuearea — run configuration",
        "=" * 44,
        f"version         : {__version__}",
        f"timestamp       : {_dt.datetime.now().isoformat(timespec='seconds')}",
        "",
        "[input / output]",
        f"input           : {args.input}",
        f"resolved slides : {len(slides)}",
        f"output dir      : {out_abs}",
        f"area table      : {os.path.join(out_abs, 'area.csv')}",
        f"area json       : {os.path.join(out_abs, 'area.json') if args.json else '(disabled)'}",
        f"thumbnails      : {os.path.join(out_abs, 'thumbnails') if save_png else '(skipped: --skip-png)'}",
        "",
        "[segmentation]",
        f"tissue type     : {args.tissue_type}",
        f"filter_grays    : {config.filter_grays}",
        f"mask_scale      : {config.mask_scale}",
        f"dilation_kernel : {config.dilation_kernel_size}",
        "",
        "[area]",
        f"headline mode   : {args.mode}",
        f"label_min_area  : {args.label_min_area} mm2",
        "",
        "slides:",
    ]
    lines += [f"  - {Path(s).name}" for s in slides]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    # Resolve inputs first, so the config banner can report the slide count.
    try:
        slides, _kind = _resolve_inputs(args.input)
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

    # Print the resolved configuration and persist it to <out>/run_config.txt.
    report = _config_report(args, slides, config, save_png)
    with open(os.path.join(out_dir, "run_config.txt"), "w") as f:
        f.write(report + "\n")
    print(report)
    print("-" * 44)

    headline_key = _AREA_MODES[args.mode]
    results = []
    failures = []
    for path in slides:
        labeled_path = (
            os.path.join(thumb_dir, f"{Path(path).stem}_regions.png") if save_png else None
        )
        try:
            out = tissue_area_for_slide(
                path,
                config=config,
                labeled_output_path=labeled_path,
                label_min_area_mm2=args.label_min_area,
            )
        except Exception as e:  # noqa: BLE001 - report and continue over a batch
            failures.append((path, str(e)))
            print(f"ERROR  {Path(path).name}: {e}", file=sys.stderr)
            continue
        results.append(out)
        if not args.quiet:
            print(
                f"{out['slide_id']}\t{args.mode}={out[headline_key]:.2f} mm²"
                f"\t(whole={out['whole_mm2']:.2f}, n_sections={out['n_sections']})"
            )

    # area.csv is always written.
    csv_path = os.path.join(out_dir, "area.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for out in results:
            row = {k: out.get(k) for k in _CSV_FIELDS}
            row["section_areas_mm2"] = ";".join(
                f"{a:.6g}" for a in out.get("section_areas_mm2", [])
            )
            writer.writerow(row)

    if args.json:
        with open(os.path.join(out_dir, "area.json"), "w") as f:
            json.dump(results, f, indent=2)

    print("-" * 44)
    print(f"Processed {len(results)}/{len(slides)} slide(s) -> {os.path.abspath(csv_path)}")
    if save_png and results:
        print(f"Thumbnails -> {os.path.abspath(thumb_dir)}")
    if failures:
        print(f"{len(failures)} slide(s) failed (see errors above).", file=sys.stderr)

    return 1 if failures and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
