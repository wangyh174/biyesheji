# Colab 运行指南（毕业设计最终完整版）

这份文档记录了如何在 Google Colab 无痛跑通整个包含"受控图像生成"、"真实样本采集"、"公平性检测打分"、"热力图可视归因"的毕业设计流水线。

## 1. 新建 Colab 并开启 GPU
- 打开一段新的笔记本
- 菜单：`Runtime (代码执行程序)` -> `Change runtime type (更改运行时类型)` -> `Hardware accelerator (硬件加速器)` 选择 `T4 GPU`。

## 2. 代码块 A：环境准备 + 下载真实样本
第一个代码块负责挂载云盘、拉取代码、安装依赖、并从 Pexels 免费图库批量下载 200 张真人医护照片作为 Real 对照组。

```python
import os
from google.colab import drive

# 1. 挂载私人云盘（弹窗出来记得点允许）
drive.mount('/content/drive')

# 2. 定义存放毕设的云盘永久避难所
workspace_dir = '/content/drive/MyDrive/Bishe'
project_dir = os.path.join(workspace_dir, 'project')
!mkdir -p {workspace_dir}

# 3. 智能同步代码库（自动检测云盘）
%cd {workspace_dir}
if not os.path.exists(project_dir):
    print("云端硬盘里还没有代码，正在克隆...")
    !git clone https://github.com/wangyh174/biyesheji.git project
else:
    print("代码库已存在，进入工作状态...")
    %cd {project_dir}
    !git pull origin main

%cd {project_dir}

# 4. 安装依赖
!pip -q install -r requirements-colab.txt
!pip -q install -e ./semantic-image-editing-main/semantic-image-editing-main

# 5. 下载真实样本（4组 × 50张 = 200张真人医护正版照片）
#    ※ 如果 data/real_samples/ 里已经有足够的图，会自动跳过不重复下载
!python scripts/download_real_samples.py \
  --per-group 50 \
  --output-dir data/real_samples \
  --pexels-key "你的Pexels_API_Key"
```

> **注意**：Pexels API Key 免费申请，30秒搞定：https://www.pexels.com/api/

## 3. 代码块 B：生成 AI 假图 + 检测器打分 + 热力图归因
第二个代码块启动 Fair-Diffusion 高画质生图（Realistic Vision 权重），然后自动跑检测器打分和热力图分析。

```python
%cd /content/drive/MyDrive/Bishe/project

# 生成 AI 假图 + 运行3个检测器 + 自动生成热力图
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

## 4. 关键成果查收
全部跑完后，毕设原始数据会自动存进你的 Google 硬盘：
- **真实样本照片：** `/MyDrive/Bishe/project/data/real_samples/`
- **AI 生成假图：** `/MyDrive/Bishe/project/data/generated_raw/`
- **检测器打分表格：** `/MyDrive/Bishe/project/results/fairness_tables/`
- **公平性汇总：** `/MyDrive/Bishe/project/results/fairness_tables/latest_run_overview.csv`
- **误判热力图（可直接插论文）：** `/MyDrive/Bishe/project/results/attribution/heatmaps/`
