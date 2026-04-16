# AIGC Fairness Evaluation and Attribution Framework

This repository represents the culmination of a graduation thesis designed to evaluate, dissect, and mitigate algorithmic bias (fairness) and generalization failures in state-of-the-art Deepfake detectors. 

The framework isolates intrinsic bias by strictly decoupling societal/environmental semantics from generative facial artifacts, evaluated across four controlled medical-worker demographic groups:
- `male-doctor`
- `female-doctor`
- `male-nurse`
- `female-nurse`

## 🌟 Core Methodological Paradigms (Thesis Innovations)

This pipeline integrates and automates several cutting-edge academic paradigms from top-tier computer vision conferences (CVPR 2024, NeurIPS 2024):

1. **Pre-generation Fairness Anchoring (Fair-Diffusion, CVPR'24)**: Fake images are generated exclusively via text-to-image semantic equalization to ensure that dataset-side biases do not pollute the detector evaluation.
2. **Strict Physical Standardization (MTCNN Face Centering)**: Before entering the AI pipeline, raw local datasets are passed through `preprocess_dataset.py` to assert a strict `512x512` physical face-anchor, completely eliminating dimension and focal-length discrepancies.
3. **Semantic Decoupling via CenterCrop (BSA, NeurIPS'24)**: To thwart shortcut learning (where models cheat by analyzing background hospitals/clothing), the pipeline invokes `CenterCrop(256)` across all detector transforms. This forces the detectors into a pure microscopic duel with the face's high-frequency pixels.
4. **Local Feature Supremacy & Demographic Disentanglement (Lin et al., CVPR'24)**: 
   - **CNNDetection**: Shows global baseline collapses across newer diffusion models.
   - **LGrad**: Validates that image gradients protect cross-domain *generalization*, but suffer from severe *demographic False Positive (FPR) bias* against specific human facial structures.
   - **NPR (Neighboring Pixel Relationships)**: Proves that micro-local statistics completely solve both Generalization and Demographic bias gaps (achieving ~0% FPR/FNR Gap parity).

---

## 🛠️ Unified Mainline Stages

The framework constructs an automated "Fairness Audit Court" broken down into the following execution stages:

- **`Stage 00`**: MTCNN raw data standardization (`preprocess_dataset.py`). Strictly standardizes authentic medical photos into `*_after` directories.
- **`Stage 01`**: Generate fake images and register local `_after` real images.
- **`Stage 01b`**: Audit generation-side imbalance before detector evaluation.
- **`Stage 02`**: Apply quality filtering, CLIP-based semantic checks, and balanced sample selection.
- **`Stage 03`**: Run pretrained detector inference for `cnndetection`, `lgrad`, and `npr`.
- **`Stage 04`**: Compute fairness metrics and stratified-bootstrap 95% confidence intervals.
- **`Stage 05`**: Produce Grad-CAM attribution maps for visual explainability.
- **`Stage 06`**: Run patch-shuffling structural attribution experiments (proving demographic bias decouple).
- **`Stage 07`**: Measure physical consistency statistics (GLCM, Second-order).
- **`Stage 08`**: Run high-pass residual experiments.
- **`Stage 09`**: Aggregate pipeline outputs into the final Markdown `MASTER_REPORT.md`.

## 🚀 Quick Start

Run the complete generalized mainline evaluation:
```bash
python scripts/00_run_local_pipeline.py --real-source local --samples 50 --buffer-extra 100 --detectors cnndetection,lgrad,npr
```

Generate fake images matching the post-MTCNN local constraints:
```bash
python scripts/01_generate.py --project-root . --real-source local --samples-per-group 150 --real-per-group 50 --model-id runwayml/stable-diffusion-v1-5 --generator fairdiffusion --seed 42
```

## ⚙️ Detector Weights Requirements

Current mainline detector checkpoints must be placed under `.external_models/weights/`:
- **CNNDetection**: `.external_models/weights/cnndetect/blur_jpg_prob0.5.pth`
- **LGrad**: `.external_models/weights/lgrad/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth`
- **NPR**: `.external_models/weights/npr/NPR.pth`

## 📈 Key Results and Data Retrieval
All evaluation outputs are perfectly categorized for thesis integration. Check the `results/` folder for:
1. `results/master_report/MASTER_REPORT.md` (Contains Final AUC and Confidence Intervals)
2. `results/structural_attribution/` (Contains the Patch-Shuffling ablation curves)
3. `results/fairness_tables/` (Contains group-by-group demographic misclassification scores)
