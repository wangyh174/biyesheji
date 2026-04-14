# Repository Update Notes

This document summarizes the **current recommended repository usage** and clarifies which files should be treated as the main path versus legacy support.

## Current recommended main path

For the real-image side, the current recommended path is:

1. Manually collect real images and place them under:
   - `data/real_samples/male-doctor/`
   - `data/real_samples/female-doctor/`
   - `data/real_samples/male-nurse/`
   - `data/real_samples/female-nurse/`
2. Run:
   - `scripts/build_real_metadata_auto.py`
   - `scripts/summarize_real_group_distribution.py`
3. Manually review `manual_keep` and `manual_note`.
4. Continue to detector inference, fairness evaluation, and attribution analysis.

## Main files to read first

If you are trying to understand the current thesis workflow quickly, read files in this order:

1. `README.md`
2. `docs/REAL_IMAGE_WORKFLOW.md`
3. `scripts/build_real_metadata_auto.py`
4. `scripts/summarize_real_group_distribution.py`
5. `scripts/02_quality_filter.py`
6. `scripts/03_run_detectors.py`
7. `scripts/04_fairness_eval.py`

## Legacy / optional files

The following file is still kept in the repository, but should be interpreted as **legacy optional support**, not the primary real-image pipeline:

- `scripts/download_real_samples.py`

Its current role is only:

- backup data collection
- candidate image expansion
- rough bootstrapping when manual search is insufficient

## Why this clarification matters

The repository originally included an automatic real-image crawler path. The current thesis workflow has shifted toward:

- manual real-image collection
- automatic metadata tagging
- group-level structural audit
- manual review on top of structured metadata

This updated path is more consistent with the thesis goal of evaluating whether detectors behave unfairly across controlled groups.
