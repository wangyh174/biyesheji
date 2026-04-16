# AIGC Fairness Evaluation and Attribution Framework

This repository contains the current thesis mainline for evaluating detector fairness across four controlled medical-worker groups:

- `male-doctor`
- `female-doctor`
- `male-nurse`
- `female-nurse`

Current mainline assumptions:

- fake images are generated with `FairDiffusion + Stable Diffusion 1.5`
- real images are manually collected under `data/real_samples/`
- Stage 02 uses fairness-sensitive CLIP filtering with profession-centered text
- current thesis detector set is `CNNDetection + LGrad + NPR`

## Current Mainline Stages

1. `Stage 01` Generate fake images and register local real images.
2. `Stage 01b` Audit generation-side imbalance before detector evaluation.
3. `Stage 02` Apply quality filtering, CLIP-based semantic checks, and balanced sample selection.
4. `Stage 03` Run pretrained detector inference for `cnndetection`, `lgrad`, and `npr`.
5. `Stage 04` Compute fairness metrics and stratified-bootstrap confidence intervals.
6. `Stage 05` Produce Grad-CAM attribution maps for the current detector set.
7. `Stage 06` Run patch-shuffling structural attribution experiments.
8. `Stage 07` Measure physical consistency statistics.
9. `Stage 08` Run high-pass residual experiments.
10. `Stage 09` Aggregate outputs into a master report.

## Mainline Notes

- `scripts/01_generate.py` is thesis-locked to `fairdiffusion` generation and `local` real-image registration.
- `scripts/02_quality_filter.py` uses profession-centered CLIP texts by default, applies conservative semantic filtering, and balances samples within each `group x y_true` slice.
- `scripts/03_run_detectors.py` keeps some legacy detector compatibility code, but the current thesis detector set is only `cnndetection,lgrad,npr`.
- `scripts/05_gradcam_analysis.py` is aligned to the same detector mainline. Legacy fallback paths are kept only for explicit compatibility use.
- `scripts/04_fairness_eval.py` estimates confidence intervals with stratified bootstrap over `group x y_true`.

## Quick Start

Run the current local mainline:

```bash
python scripts/00_run_local_pipeline.py --real-source local --samples 50 --buffer-extra 100 --detectors cnndetection,lgrad,npr
```

Generate fake images and register existing local real images:

```bash
python scripts/01_generate.py --project-root . --real-source local --samples-per-group 150 --real-per-group 50 --model-id runwayml/stable-diffusion-v1-5 --generator fairdiffusion --seed 42
```

Run Stage 02 filtering:

```bash
python scripts/02_quality_filter.py --project-root . --metadata-in data/metadata_raw.csv --metadata-out data/metadata_balanced.csv --balanced-dir data/generated_balanced --use-clip --clip-text-mode profession --align-on clip --copy-files
```

Important outputs are written under `data/` and `results/`.

## Detector Weights

Current mainline detector checkpoints:

- `CNNDetection`
- Official repo: `https://github.com/PeterWang512/CNNDetection`
- Expected local checkpoint: `.external_models/weights/cnndetect/blur_jpg_prob0.5.pth`

- `LGrad`
- Official repo: `https://github.com/chuangchuangtan/LGrad`
- Supported checkpoint filenames:
- `LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth`
- `LGrad-2class-Trainon-Progan_chair_horse.pth`
- `LGrad-1class-Trainon-Progan_horse.pth`
- Expected local directory: `.external_models/weights/lgrad/`

- `NPR`
- Official repo: `https://github.com/chuangchuangtan/NPR-DeepfakeDetection`
- Expected local checkpoint: `.external_models/weights/npr/NPR.pth`

## Practical Advice

- Keep manually collected real images under `data/real_samples/<group>/`.
- Use `--real-per-group` to cap how many local real images per group are registered in Stage 01.
- Over-generate fake candidates, then let Stage 02 perform conservative filtering and equalized retention.
- Inspect `results/fairness_tables/quality_clip_summary_before.csv`, `quality_clip_summary_after.csv`, and `quality_clip_filter_audit.csv` after Stage 02 to verify that post-processing has not disproportionately removed one group.
- If a group cannot reach the requested count, Stage 02 will cap balancing to the smallest surviving group.
