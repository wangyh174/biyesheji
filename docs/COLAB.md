# Google Colab 运行说明

这份文档用于在 Google Colab 上运行本项目，并把下面三类关键数据持久化到 Google Drive：

- `data/real_samples`
- `data/generated_raw`
- `results`

## 1. 挂载 Google Drive 并拉取项目

```python
from google.colab import drive
drive.mount("/content/drive")

%cd /content
!rm -rf project
!git clone https://github.com/wangyh174/biyesheji.git project
%cd /content/project
!git pull origin main
```

## 2. 安装依赖

先安装通用依赖：

```python
!pip install -U pip
!pip install -r requirements.txt
```

再单独安装 `fairdiffusion` 需要的本地包：

```python
!pip install --no-build-isolation ./semantic-image-editing-main/semantic-image-editing-main
```

说明：

- 不再推荐使用 `pip install -e ...`，因为在 Colab 上这类老式 `setup.py` 包做 editable install 容易失败。
- 现在推荐的方式是普通安装，并显式加 `--no-build-isolation`。

## 3. 把关键目录映射到 Google Drive

```python
import os
import shutil
from pathlib import Path

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

print("已映射到 Google Drive：")
print("/content/project/data/real_samples ->", drive_root / "data/real_samples")
print("/content/project/data/generated_raw ->", drive_root / "data/generated_raw")
print("/content/project/results ->", drive_root / "results")
```

## 4. 采集真人图

```python
%cd /content/project
!python scripts/download_real_samples.py \
    --samples-per-group 60 \
    --clip-threshold 0.22
```

说明：

- 当前会优先从这些来源抓图：
  - `Unsplash`
  - `Pexels`
  - `Openverse`
  - `Pixabay`
- 每个 group 目录下还会额外生成：
  - `_candidates/`
  - `_candidate_scores.csv`

如果正式通过的图不够，就从 `_candidates/` 里人工补一些明显正确的图。

## 5. 运行完整流水线

```python
%cd /content/project
!python scripts/00_run_local_pipeline.py \
    --real-source local \
    --samples 50 \
    --detectors cnndetection,f3net,gram,lgrad
```

## 6. 如果只想单独检查 Grad-CAM

```python
%cd /content/project
!python scripts/05_gradcam_analysis.py \
    --detector-csv results/detector_outputs/cnndetection_scores.csv \
    --analyze-all \
    --max-per-group 2
```

## 7. 下载结果

因为 `data/real_samples`、`data/generated_raw`、`results` 已经映射到 Google Drive，所以这些文件会自动持久化保存。

如果还想额外下载一份压缩包到本地：

```python
%cd /content/project
!zip -r result_v2_N50_final.zip \
    results/ \
    data/generated_raw/ \
    data/real_samples/
```

```python
from google.colab import files
files.download("result_v2_N50_final.zip")
```
