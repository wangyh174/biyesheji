# Google Colab 实验执行指南 (Full Stages 01-09)

本指南针对 AIGC 检测公平性评估的全流程，包括高精度的受控生成、公平性评估及深度归因分析。

## ⚙️ 第一部分：环境准备与驱动挂载

```python
# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 克隆项目并安装依赖 (包含 Fair-Diffusion 核心库)
%cd /content
!git clone https://github.com/wangyh174/biyesheji.git project
%cd /content/project
!pip install -r requirements.txt
!pip install -e semantic-image-editing-main/
!pip install opencv-python scikit-image transformers diffusers
```

## 🛠️ 第二部分：全流程流水线执行 (01-09)

直接运行总控脚本，即可完成生成、过滤、评估及全部归因实验（Grad-CAM+Patch Shuffling+GLCM）：

```python
# 执行完整的毕设实验路径 (Fair-Diffusion + 多级归因分析)
# 指定 --real-source local 使用已下载的真实样本 ( data/real_samples/ )
!python scripts/00_run_local_pipeline.py \
    --real-source local \
    --detectors cnndetection,f3net \
    --samples 50 \
    --model-id "SG161222/Realistic_Vision_V5.1_noVAE"
```

## 🔍 第三部分：关键归因实验单独执行 (进阶)

如果你想单独运行特定的归因实验，可以使用以下命令：

### 1. Patch Shuffling (结构性归因 - NeurIPS 24)
打乱语义结构，验证偏见随语义消失而消解的趋势：
```python
!python scripts/06_patch_shuffling_exp.py --patch-n 8
# 在打乱后的数据上重跑检测
!python scripts/03_run_detectors.py --detector f3net --input-dir data/shuffled_8x8
```

### 2. 物理一致性验证 (二阶统计 - ICCV 25)
通过 GLCM 特征证明 AI 物理痕迹在跨性别/职业下的公平性：
```python
!python scripts/07_physical_consistency.py --max-samples 50
```

### 3. 解耦创新方案 (高通滤波器 - Innovation)
展示基于物理解耦的解偏见架构：
```python
!python scripts/08_high_pass_innovation.py
!python scripts/03_run_detectors.py --detector cnndetection --input-dir data/high_pass_residuals
```

## 📊 第四部分：结果分析与可视化

*   **热力图**：查看 `results/gradcam/` 目录。
*   **公平性指标表**：查看 `results/fairness_tables/`。
*   **总报告汇总**：查看 `results/master_report.csv`。

> [!TIP]
> 建议在生成 Fake 样本前，先运行 `scripts/download_real_samples.py` 补充 Real 样本池，以确保 FPR Gap 的计算基准更稳定。
