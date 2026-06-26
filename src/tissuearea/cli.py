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
import sys
from typing import List, Optional

from . import __version__
from .area import tissue_area_for_slide
from .config import MaskingConfig

# Per-slide fields written to CSV / printed (excludes the variable-length
# section list).
_SCALAR_FIELDS = [
    "slide_id",
    "whole_mm2",
    "largest_cc_mm2",
    "top2_sum_mm2",
    "n_sections",
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
    p.add_argument("--csv", metavar="PATH", help="Write scalar per-slide results as CSV.")
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

    headline_key = _AREA_MODES[args.mode]
    results = []
    failures = []

    for path in args.slides:
        try:
            out = tissue_area_for_slide(path, config=config)
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
            writer = csv.DictWriter(f, fieldnames=_SCALAR_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for out in results:
                writer.writerow(out)
        if not args.quiet:
            print(f"Wrote {len(results)} rows -> {args.csv}", file=sys.stderr)

    return 1 if failures and not results else 0


if __name__ == "__main__":
    raise SystemExit(main())
