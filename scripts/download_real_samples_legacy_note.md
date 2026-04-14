# Legacy Note for `download_real_samples.py`

`scripts/download_real_samples.py` is kept in the repository as a **legacy optional crawler**.

It is **not** the recommended main workflow for current thesis experiments.

## Current recommended real-image workflow

Please use:

1. Manual real-image collection (for example via Google search)
2. `scripts/build_real_metadata_auto.py`
3. `scripts/summarize_real_group_distribution.py`
4. manual review through `manual_keep` and `manual_note`

## Why the crawler is no longer primary

The thesis focuses on fairness evaluation across controlled groups. The current preferred workflow puts more emphasis on:

- stable group labels
- controlled face scale distribution
- controlled scene distribution
- controlled clothing cue distribution
- structural audit before detector evaluation

The old crawler can still be useful for backup collection or candidate expansion, but it should be treated as a secondary utility rather than the default path.
