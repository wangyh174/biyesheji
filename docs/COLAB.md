# Colab Run Guide

This guide assumes you want to run the thesis pipeline in Google Colab and keep key data in Google Drive.

## 1. Mount Drive and clone the project

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!rm -rf project
!git clone <YOUR_GITHUB_REPO_URL> project
%cd /content/project
```

## 2. Install dependencies

```python
!pip install -U pip
!pip install torch torchvision torchaudio
!pip install diffusers transformers accelerate safetensors opencv-python scikit-image tabulate gdown
```

If you use Fair-Diffusion mode, also install the local semantic editing package described by the repository.

## 3. Persist key folders to Drive

```python
import os
from pathlib import Path
import shutil

drive_root = Path("/content/drive/MyDrive/bishe_project_runtime")
for rel in ["data/real_samples", "data/generated_raw", "results"]:
    target = drive_root / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.mkdir(parents=True, exist_ok=True)
    local = Path("/content/project") / rel
    if local.is_symlink():
        local.unlink()
    elif local.exists():
        if local.is_dir():
            shutil.rmtree(local)
        else:
            local.unlink()
    local.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(target, local, target_is_directory=True)

print("Mapped to Drive:")
print("/content/project/data/real_samples ->", drive_root / "data/real_samples")
print("/content/project/data/generated_raw ->", drive_root / "data/generated_raw")
print("/content/project/results ->", drive_root / "results")
```

## 4. Collect real images

```python
%cd /content/project
!python scripts/download_real_samples.py \
    --samples-per-group 60 \
    --clip-threshold 0.22
```

Notes:

- The crawler now tries to reject dolls, figurines, cartoons, mannequins, and object photos.
- If a group still contains wrong samples, delete that group folder under `data/real_samples/<group>` and rerun the crawler.

## 5. Run the full pipeline

```python
%cd /content/project
!python scripts/00_run_local_pipeline.py \
    --real-source local \
    --samples 50 \
    --detectors cnndetection,f3net,gram,lgrad
```

Current pipeline behavior:

- Stage 01 over-generates `samples + 20` candidates per group for better headroom.
- `scripts/01_generate.py` now uses Fair-Diffusion-style generation more faithfully: for `fairdiffusion` mode it keeps the occupation prompt concise and controls gender mainly through editing directions, while still keeping stricter negative prompts as a quality safeguard.
- In `fairdiffusion` mode, direction choice is now sampled with a target prior and each generated image is saved into the sampled gender-profession bucket. Use `--fd-female-prob 0.5` to mimic the paper's balanced gender sampling more closely.
- Stage 02 uses CLIP prompt similarity plus:
  - target-group prediction agreement
  - group margin threshold
  - human-photo preference over toy/cartoon/object/deformed/low-quality prompts
- `scripts/03_run_detectors.py` uses pretrained detector inference.
- `scripts/04_fairness_eval.py` uses stratified bootstrap on `group × y_true`.
- `scripts/05_gradcam_analysis.py` uses real model Grad-CAM.

## 6. Optional: test Grad-CAM separately

```python
%cd /content/project
!python scripts/05_gradcam_analysis.py \
    --detector-csv results/detector_outputs/cnndetection_scores.csv \
    --analyze-all \
    --max-per-group 2
```

## 7. Common quality problems

- `male-doctor` images look female:
  Stage 01 prompt drift or Stage 02 semantic filtering is too loose. Rerun after deleting bad generated files.
- Faces are broken or melted:
  The generation model produced low-quality candidates; Stage 02 should remove most of them, but you may need to over-generate again.
- A group has too few remaining samples:
  That group became the balancing bottleneck. Generate or collect more candidates for that group only, then rerun Stage 02 onward.

## 8. Export results

```python
%cd /content/project
!zip -r result_v2_N50_final.zip results/ data/high_pass_residuals/ data/physical_consistency_results.csv
```

Because `results/` is already mapped to Drive, your main outputs are persistent even if the Colab runtime disconnects.
