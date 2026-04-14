# Real Image Workflow

This document describes the **current recommended real-image workflow** for this project.

## Recommended workflow

The recommended real-image pipeline is now:

1. Manually collect real images for the four groups, for example via Google search.
2. Organize the images under:
   - `data/real_samples/male-doctor/`
   - `data/real_samples/female-doctor/`
   - `data/real_samples/male-nurse/`
   - `data/real_samples/female-nurse/`
3. Run automatic metadata construction:

```bash
python scripts/build_real_metadata_auto.py
```

4. Run group-level distribution summaries:

```bash
python scripts/summarize_real_group_distribution.py
```

5. Manually review the generated CSV, fill `manual_keep` and `manual_note`, and then rerun:

```bash
python scripts/summarize_real_group_distribution.py --only-kept
```

## Why this is now preferred

The thesis focuses on whether detectors behave unfairly across controlled groups, so the real-image workflow should prioritize:

- stable group labels
- face scale comparability
- scene comparability
- clothing cue comparability
- image quality checks
- duplicate screening

The manual-collection + automatic-audit workflow is better aligned with this goal than a purely automated crawler.

## Legacy crawler note

`scripts/download_real_samples.py` is retained as a **legacy optional crawler**.

It may still be useful for:

- backup data collection
- candidate image expansion
- rough bootstrapping when manual search is not enough

However, it is **no longer the recommended main workflow** for real-image construction in this repository.

## Suggested repository reading order

If you want to understand the current real-image path first, read files in this order:

1. `docs/REAL_IMAGE_WORKFLOW.md`
2. `scripts/build_real_metadata_auto.py`
3. `scripts/summarize_real_group_distribution.py`
4. `scripts/02_quality_filter.py`
