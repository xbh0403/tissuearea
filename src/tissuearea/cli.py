"""Command-line interface: estimate tissue area (mm²) for one or more slides.

Examples
--------
    tissuearea slide.svs
    tissuearea *.svs --no-grays --mode largest_cc
    tissuearea slides/*.svs --no-grays --csv areas.csv
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .area import tissue_area_for_slide
from .config import MaskingConfig

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
    p.add_argument("slides", nargs="+", help="WSI paths (.svs, .ndpi, .tiff, ...).")
    p.add_argument(
        "--mode",
        choices=sorted(_AREA_MODES),
        default="largest_cc",
        help="Which area candidate to print as the headline value (default: largest_cc).",
    )
    p.add_argument(
        "--no-grays",
        action="store_true",
        help="Disable the gray filter (filter_grays=False); recommended for faint "
        "fresh-frozen tissue.",
    )
    p.add_argument(
        "--scale",
        type=int,
        default=None,
        help="Thumbnail downsampling factor (default: MaskingConfig.mask_scale=32).",
    )
    p.add_argument("--json", metavar="PATH", help="Write full per-slide results as JSON.")
    p.add_argument(
        "--csv",
        metavar="PATH",
        help="Write per-slide results as CSV (total, largest, and all region areas).",
    )
    p.add_argument(
        "--save-labeled",
        metavar="DIR",
        help="Save an annotated thumbnail per slide ({slide_id}_regions.png) with "
        "each region outlined and its area (mm²) labelled.",
    )
    p.add_argument(
        "--label-min-area",
        type=float,
        default=0.0,
        metavar="MM2",
        help="In labelled thumbnails, only write area text for regions >= this many "
        "mm² (contours are still drawn). Default 0 = label every region.",
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress the per-slide stdout table."
    )
    p.add_argument("--version", action="version", version=f"tissuearea {__version__}")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    cfg_kwargs = {}
    if args.no_grays:
        cfg_kwargs["filter_grays"] = False
    if args.scale is not None:
        cfg_kwargs["mask_scale"] = args.scale
    config = MaskingConfig(**cfg_kwargs)

    if args.save_labeled:
        os.makedirs(args.save_labeled, exist_ok=True)

    headline_key = _AREA_MODES[args.mode]
    results = []
    failures = []

    for path in args.slides:
        labeled_path = (
            os.path.join(args.save_labeled, f"{Path(path).stem}_regions.png")
            if args.save_labeled
            else None
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
            print(f"ERROR  {path}: {e}", file=sys.stderr)
            continue
        results.append(out)
        if not args.quiet:
            print(
                f"{out['slide_id']}\t{args.mode}={out[headline_key]:.2f} mm²"
                f"\t(whole={out['whole_mm2']:.2f}, n_sections={out['n_sections']})"
            )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        if not args.quiet:
            print(f"Wrote {len(results)} records -> {args.json}", file=sys.stderr)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for out in results:
                row = {k: out.get(k) for k in _CSV_FIELDS}
                row["section_areas_mm2"] = ";".join(
                    f"{a:.6g}" for a in out.get("section_areas_mm2", [])
                )
                writer.writerow(row)
        if not args.quiet:
            print(f"Wrote {len(results)} rows -> {args.csv}", file=sys.stderr)

    if args.save_labeled and not args.quiet:
        print(
            f"Saved {len(results)} labelled thumbnail(s) -> {args.save_labeled}",
            file=sys.stderr,
        )

    return 1 if failures and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
