# Google Colab 运行说明

这份文档用于在 Google Colab 上运行本项目，并把下面三类关键数据持久化到 Google Drive：

- `data/real_samples`
- `data/generated_raw`
- `results`

这样即使 Colab 断开，已经采集的真人图、已经生成的 fake 图、以及实验结果都不会丢失。

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

```python
!pip install -U pip
!pip install torch torchvision torchaudio
!pip install diffusers transformers accelerate safetensors opencv-python scikit-image tabulate gdown
!pip install -e semantic-image-editing-main/semantic-image-editing-main
```

其中最后一条用于安装 `fairdiffusion` 所需的 `semdiffusers`。

## 3. 把关键目录映射到 Google Drive

下面这段代码会把：

- `/content/project/data/real_samples`
- `/content/project/data/generated_raw`
- `/content/project/results`

映射到 Google Drive 里的：

- `/content/drive/MyDrive/bishe_project_runtime/data/real_samples`
- `/content/drive/MyDrive/bishe_project_runtime/data/generated_raw`
- `/content/drive/MyDrive/bishe_project_runtime/results`

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

- 当前脚本会优先从这些来源抓图：
  - `Unsplash`
  - `Pexels`
  - `Openverse`
  - `Pixabay`
- 已内置 `Pexels`、`Pixabay`、`Unsplash` 调用逻辑。
- 现在爬虫已经增加了真人照片过滤，会尽量排除：
  - 玩偶
  - 卡通
  - 摆件
  - 雕像
  - 普通非医疗人物肖像

### 新的半自动候选复查机制

现在每个 group 目录下除了正式通过的图片外，还会额外生成：

- `_candidates/`
- `_candidate_scores.csv`

含义如下：

- `正式通过的图片`
  直接保存在 `data/real_samples/<group>/`
- `未正式通过但分数较高的候选图`
  保存在 `data/real_samples/<group>/_candidates/`
- `每张候选图的评分明细`
  写在 `data/real_samples/<group>/_candidate_scores.csv`

`_candidate_scores.csv` 里会包含这些字段：

- `human_prob`
- `target_prob`
- `competitor_best`
- `negative_best`
- `final_score`
- `passed`

如果某个 group 最终数量不够，不要立刻继续调阈值。  
先去对应目录下看：

- `data/real_samples/male-doctor/_candidates/`
- `data/real_samples/male-doctor/_candidate_scores.csv`

把明显正确的真人医护图人工挑一些补到 group 根目录里即可。

这一步非常适合毕设，因为你只需要每组几十张干净图，不需要把整个采集过程做到完全无人值守。

## 5. 运行完整流水线

```python
%cd /content/project
!python scripts/00_run_local_pipeline.py \
    --real-source local \
    --samples 50 \
    --detectors cnndetection,f3net,gram,lgrad
```

当前流水线的关键行为：

- Stage 01 会先多生成 `samples + 20` 张 fake 候选图，再交给后续筛选
- `scripts/01_generate.py` 已更贴近 Fair-Diffusion：
  - `fairdiffusion` 模式下主 prompt 更简洁
  - 性别主要通过 editing direction 控制
  - direction 按目标先验采样
  - 可用 `--fd-female-prob 0.5` 接近论文里的平衡采样
- `scripts/02_quality_filter.py` 会过滤：
  - 群体不一致图
  - 更像玩偶/卡通/物体的图
  - 畸形脸和低质量图
- `scripts/03_run_detectors.py` 已切换为预训练检测器推理
- `scripts/04_fairness_eval.py` 已改为 `group × y_true` 分层 bootstrap
- `scripts/05_gradcam_analysis.py` 已改为真实模型 Grad-CAM

## 6. 如果只想单独检查 Grad-CAM

```python
%cd /content/project
!python scripts/05_gradcam_analysis.py \
    --detector-csv results/detector_outputs/cnndetection_scores.csv \
    --analyze-all \
    --max-per-group 2
```

## 7. 常见问题

- `male-doctor` 里混入女性  
  说明生成阶段性别漂移，或者 Stage 02 过滤不够严格。删掉坏的 fake 图后重新跑 Stage 01 和 Stage 02。

- fake 图脸崩、融化、眼睛不对  
  这是生成模型候选质量问题。现在流程会尽量在 Stage 02 过滤掉，但如果坏图太多，还是建议重新多生成一轮。

- 真人图某个组数量不够  
  不要立刻再改阈值。先看该组的：
  - `_candidates/`
  - `_candidate_scores.csv`
  再人工补图。

## 8. 把结果保存到 Google Drive 并下载到本地

因为 `data/real_samples`、`data/generated_raw`、`results` 已经映射到了 Google Drive，所以运行过程中这些内容会自动持久化保存。

如果你还想额外打包一份结果下载到本地，可以执行：

```python
%cd /content/project
!zip -r result_v2_N50_final.zip \
    results/ \
    data/generated_raw/ \
    data/real_samples/
```

然后下载到本地：

```python
from google.colab import files
files.download("result_v2_N50_final.zip")
```

如果你还想在 Google Drive 里单独再留一份压缩包，也可以执行：

```python
backup_zip = "/content/drive/MyDrive/bishe_project_runtime/result_v2_N50_final.zip"
!cp result_v2_N50_final.zip {backup_zip}
print("压缩包已保存到:", backup_zip)
```
