# tissuearea

Estimate the **physical tissue area (mm²)** on a whole slide image (WSI) by
combining a tissue segmentation mask with the slide's microns-per-pixel (MPP)
metadata.

`tissuearea` is a small, self-contained library: give it a slide (or a
thumbnail + dimensions + MPP) and it returns both the **whole-slide** tissue
area and the **largest single section** area, plus the full list of per-section
(connected-component) areas. It has no machine-learning dependencies — just
OpenSlide, OpenCV, scikit-image, and NumPy.

> The segmentation primitives and area math are extracted, dependency-free, from
> the EasyMSI preprocessing pipeline; a default-config mask reproduces the EasyMSI
> production segmentation exactly.

## How it works

A scale-`s` thumbnail maps linearly onto level 0, so each mask pixel covers
`(W·mpp_x / mask_w) · (H·mpp_y / mask_h)` µm², where `W, H` are the level-0 slide
dimensions and `mask_w, mask_h` the thumbnail dimensions.

1. **Segment** the thumbnail: CLAHE + Otsu → pen/gray color filters →
   remove-small-objects → fill-holes → optional dilation
   (`tissuearea.build_tissue_mask`).
2. **Convert** the boolean mask to area, reporting candidates
   (`tissuearea.area_from_mask`):
   - `whole_mm2` — all detected tissue
   - `largest_cc_mm2` — the single largest connected component (one section)
   - `top2_sum_mm2`, `n_sections`, and the sorted `section_areas_mm2`

Connected components use 8-connectivity. MPP resolution order:
`openslide.mpp-x/-y` → an explicit fallback → `10 / objective-power`.

## Install

### 1. Create & activate a conda env

```bash
conda create -n tissuearea python=3.12 -y
conda activate tissuearea
```

### 2. pip-install OpenSlide binaries + the package

```bash
# OpenSlide C library via pip — no system/conda install needed:
pip install openslide-bin

pip install -e .            # from this repo
pip install -e ".[dev]"     # with pytest
```

`openslide-python` and the other dependencies are pulled in automatically;
`openslide-bin` supplies the native OpenSlide library it binds to.

## Usage

### Command line

```bash
tissuearea slide.svs                                   # fresh-frozen (default)
tissuearea slides/*.svs --mode largest_cc --csv areas.csv
tissuearea slides/*.svs --type ffpe --csv areas.csv    # FFPE cohort
tissuearea slide.svs --json out.json
# also save an annotated thumbnail per slide (regions outlined + area-labelled):
tissuearea slides/*.svs --csv areas.csv --save-labeled labels/
```

```
M1011007  largest_cc=37.25 mm²  (whole=74.40, n_sections=6)
```

**`--type`** picks the segmentation tuned for the tissue preparation:
`ff` = fresh-frozen (**default** — gray filter off, keeps pale frozen tissue) or
`ffpe` = FFPE (gray filter on). Frozen tissue is often near-neutral, so the gray
filter would discard it (sometimes yielding an empty mask); FFPE is well-stained,
so the filter is kept on. Set the one matching your cohort.

**`--csv` columns** — one row per slide:

| column | meaning |
|---|---|
| `whole_mm2` | total tissue area (all regions) |
| `largest_cc_mm2` | largest single region (one section) |
| `section_areas_mm2` | **every** region's area, mm², largest-first, `;`-separated |
| `top2_sum_mm2`, `n_sections`, `mask_fraction` | summary |
| `width`, `height`, `mpp_x`, `mpp_y`, `mask_scale` | slide metadata |

`--save-labeled DIR` writes `{slide_id}_regions.png` per slide — each connected
tissue region outlined and labelled with its area (`#1` = largest), plus a header
showing the region count and total/largest area. Use `--label-min-area MM2` to
suppress text on tiny specks (contours are still drawn). The full JSON (`--json`)
additionally carries the raw `section_areas_mm2` list per slide.

### Python

```python
from tissuearea import tissue_area_for_slide, masking_config_for_type

# Pick the segmentation for your tissue type: "ff" (fresh-frozen) or "ffpe".
cfg = masking_config_for_type("ff")
out = tissue_area_for_slide("slide.svs", cfg, labeled_output_path="slide_regions.png")
print(out["largest_cc_mm2"], out["section_areas_mm2"])  # largest, and all regions

# Bare MaskingConfig() is the production default (gray filter ON = FFPE-style).
```

Work directly from a thumbnail you already have:

```python
import numpy as np
from tissuearea import build_tissue_mask, area_from_mask, region_areas, draw_region_labels

mask = build_tissue_mask(thumbnail_rgb)
area = area_from_mask(mask, width=W, height=H, mpp_x=0.2628, mpp_y=0.2628)
regions = region_areas(mask, W, H, 0.2628, 0.2628)   # [{rank, area_mm2, centroid_xy, ...}, ...]
draw_region_labels(thumbnail_rgb, mask, W, H, 0.2628, 0.2628, output_path="regions.png")
```

## Public API

| Symbol | Purpose |
|---|---|
| `tissue_area_for_slide(path, config=None)` | Open a slide read-only and return all area candidates + metadata. |
| `tissue_area_from_thumbnail(thumb, W, H, mpp_x, mpp_y, config=None)` | Segment a thumbnail and compute areas. |
| `build_tissue_mask(thumb, config=None)` | RGB thumbnail → boolean tissue mask (= the production segmentation). |
| `area_from_mask(mask, W, H, mpp_x, mpp_y)` | Boolean mask → area dict (whole / largest_cc / top2 / sections). |
| `region_areas(mask, W, H, mpp_x, mpp_y)` | Per-region list (rank, area_mm2, centroid, bbox), largest-first. |
| `draw_region_labels(thumb, mask, W, H, mpp_x, mpp_y, output_path=None, min_area_mm2=0.0)` | Annotated thumbnail: each region outlined + area-labelled. |
| `resolve_mpp_xy(props, fallback_mpp=None)` | Resolve `(mpp_x, mpp_y)` from OpenSlide properties. |
| `visualize_mask(thumb, mask, output_path=None)` | Black out non-tissue for QA. |
| `SlideReader(path)` | Minimal read-only OpenSlide wrapper (`dimensions`, `mpp`, `properties`, `get_thumbnail`). |
| `MaskingConfig` | Segmentation parameters (the defaults are production; pass overrides). |
| `masking_config_for_type("ff"\|"ffpe", **overrides)` | Optimal `MaskingConfig` for a tissue type (what the CLI `--type` uses). |

## Choosing whole-slide vs. largest-section area

On a validation set of 307 fresh-frozen colorectal slides (reference =
single-section areas), the best agreement came from **disabling the gray filter
and taking the largest connected component**:

| Config (area mode) | Spearman | median |%err| | within ±25% |
|---|---|---|---|
| `filter_grays=False`, `largest_cc` | **0.90** | **9.9%** | **70.7%** |
| `filter_grays=False`, `whole` | 0.84 | 14.5% | — |
| default, `whole` | 0.76 | 20.3% | 50.8% |

Two practical takeaways:

- **Use `largest_cc_mm2`** when the reference measures one tissue section; a slide
  with two sections otherwise over-predicts ~2× on `whole_mm2`.
- **Match the tissue type** (`--type ff` / `masking_config_for_type("ff")`) for
  faint, near-neutral fresh-frozen tissue — the gray filter otherwise discards it
  and can even yield empty masks. This is baked into the `ff` preset (the CLI
  default). FFPE (`--type ffpe`) keeps the filter on. Higher mask resolution
  (smaller `mask_scale`) is *not* the lever; the gray filter is.

## Testing

```bash
pytest
```

The area math is pinned against hand-computed values on synthetic masks; the
masking tests check the pipeline on a synthetic H&E-like thumbnail (blob
detected, background rejected, gray-filter toggle behaves monotonically). No
real `.svs` is required.

## License

MIT
