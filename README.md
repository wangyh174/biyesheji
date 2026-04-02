# AIGC Fairness Evaluation and Attribution Framework

This project is a thesis-oriented pipeline for evaluating whether AI-image detectors behave unfairly across four controlled medical-worker groups:

- `male-doctor`
- `female-doctor`
- `male-nurse`
- `female-nurse`

The pipeline combines controlled image generation, real-image collection, semantic and quality filtering, detector evaluation, fairness metrics, and attribution analysis.

## Stages

1. `Stage 01` Generate fake images and organize real images by group.
2. `Stage 01b` Audit generation-side imbalance before detector evaluation.
3. `Stage 02` Filter samples with CLIP, quality control, and group-consistency checks.
4. `Stage 03` Run pretrained detector inference for `CNNDetection`, `F3Net`, `Gram`, and `LGrad`.
5. `Stage 04` Compute fairness metrics and stratified-bootstrap confidence intervals.
6. `Stage 05` Produce model-based Grad-CAM heatmaps.
7. `Stage 06` Run patch-shuffling structural attribution experiments.
8. `Stage 07` Measure second-order physical consistency statistics.
9. `Stage 08` Run high-pass residual decoupling experiments.
10. `Stage 09` Aggregate results into a master report.

## Current Implementation Notes

- `scripts/03_run_detectors.py` uses pretrained detector inference instead of the old handcrafted-feature + LogisticRegression proxy.
- `scripts/04_fairness_eval.py` estimates confidence intervals with stratified bootstrap over `group x y_true`.
- `scripts/05_gradcam_analysis.py` uses true model-backpropagation Grad-CAM rather than residual pseudo-heatmaps.
- `scripts/01_generate.py` uses two behaviors: in normal diffusion mode it strengthens prompts for `male/female x doctor/nurse`, while in `fairdiffusion` mode it keeps the occupation prompt concise and shifts gender control into Fair-Diffusion-style editing directions.
- In `fairdiffusion` mode, gender directions are sampled probabilistically per profession and images are routed into the corresponding `male-*` / `female-*` groups. The sampling prior can be controlled by `--fd-female-prob`.
- `scripts/02_quality_filter.py` checks more than prompt similarity: it also verifies target-group consistency, group margin, and whether an image is more like a real human photo than a toy, cartoon, object, deformed face, or low-quality image.
- `scripts/download_real_samples.py` now supports Google Programmable Search in addition to Unsplash, Pexels, Openverse, and Pixabay, and prioritizes single-person real-photo queries.
- Real-image crawling keeps both accepted images and a candidate-review pool under `_candidates/` with a `_candidate_scores.csv` file for manual review.
- In Colab, it is recommended to map `data/real_samples`, `data/generated_raw`, and `results` into Google Drive for persistence.

## Quick Start

Run the full local pipeline:

```bash
python scripts/00_run_local_pipeline.py --real-source local --detectors cnndetection,f3net,gram,lgrad
```

Crawl real images with Google-enabled search (leave the Google values empty until you have them):

```bash
set GOOGLE_API_KEY=
set GOOGLE_CSE_ID=
python scripts/download_real_samples.py --samples-per-group 100 --clip-threshold 0.22
```

Important outputs are written under `data/` and `results/`.

## Practical Advice

- If real-image crawling or fake-image generation produces poor samples, delete the bad group folder and rerun Stage 01 or the specific script.
- For best fake-image quality, over-generate candidates and let Stage 02 keep only the most semantically consistent samples.
- If a group cannot reach the requested sample count automatically, review `_candidates/` and `_candidate_scores.csv`, then manually move correct images into the group folder.
- If a group still cannot reach the requested count, Stage 02 will cap balancing to the smallest surviving group.
