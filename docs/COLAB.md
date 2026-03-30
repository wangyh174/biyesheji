# Colab 运行指南（毕业设计最终完整版）

## 1. 新建 Colab 并开启 GPU
- 菜单：`Runtime` -> `Change runtime type` -> `T4 GPU`

## 2. 代码块 A：环境准备 + 下载真实样本（带 CLIP 智能筛选）

```python
import os
from google.colab import drive

drive.mount('/content/drive')
workspace_dir = '/content/drive/MyDrive/Bishe'
project_dir = os.path.join(workspace_dir, 'project')
!mkdir -p {workspace_dir}

%cd {workspace_dir}
if not os.path.exists(project_dir):
    !git clone https://github.com/wangyh174/biyesheji.git project
else:
    %cd {project_dir}
    !git pull origin main

%cd {project_dir}

!pip -q install -r requirements-colab.txt
!pip -q install -e ./semantic-image-editing-main/semantic-image-editing-main

# 清掉旧的低质量图（首次运行可忽略）
!rm -rf data/real_samples/*

# 下载真人照片，CLIP 自动剔除不匹配的图（如男护士文件夹里的女医生）
!python scripts/download_real_samples.py \
  --per-group 50 \
  --output-dir data/real_samples \
  --pexels-key "你的Pexels_API_Key"
```

## 3. 代码块 B：生成 AI 假图 + 检测器打分 + 热力图

```python
%cd /content/drive/MyDrive/Bishe/project

!python scripts/00_run_local_pipeline.py \
  --project-root ./ \
  --generator fairdiffusion \
  --samples-per-group 50 \
  --real-per-group 50 \
  --detectors cnndetection,f3net,lgrad \
  --clip-min-score 0.22 \
  --real-source diffusers \
  --model-id SG161222/Realistic_Vision_V5.1_noVAE \
  --real-model-id SG161222/Realistic_Vision_V5.1_noVAE
```

## 4. 成果查收
- **真实样本：** `data/real_samples/`（4组×50张，CLIP验证过）
- **AI假图：** `data/generated_raw/`
- **公平性表格：** `results/fairness_tables/latest_run_overview.csv`
- **热力图：** `results/attribution/heatmaps/`
