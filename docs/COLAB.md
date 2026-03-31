# Google Colab 毕设实验终极执行指南 (Stage 01-09 Bugfix 版)

本指南针对 AIGC 检测公平性评估全流程 (n=50)。

## ⚙️ 第一部分：环境配置与驱动挂载
建议使用 **L4 GPU** 或 **A100 GPU** 运行。

```python
# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 拉取毕设框架 (Stage 01-09 完整版)
%cd /content
!rm -rf project 
!git clone https://github.com/wangyh174/biyesheji.git project
%cd /content/project
!git pull origin main

# 3. 安装依赖 (已修复嵌套路径与 requirements.txt 报错)
!pip install -r requirements.txt
!pip install -e semantic-image-editing-main/semantic-image-editing-main/
!pip install opencv-python scikit-image transformers diffusers accelerate
```

## 🛠️ 第二部分：双库精选真人样本采集 (n=50)
利用“语义解歧”机制，过滤掉由于标签重叠产生的“像医生的护士”。
```python
# 采集 200 张 (50x4) 经过 CLIP 语义解歧校准的真人对照图
# 已内置 Pexels + Pixabay 双 Key，无需填写
!python scripts/download_real_samples.py \
    --samples-per-group 50 \
    --clip-threshold 0.22
```

## 🚀 第三部分：执行全量毕设管线 (01-09)
包含生成、基准评价、视觉/结构/物理归因及解耦创新。
```python
# 开启 50 规模的自动化深度评估
!python scripts/00_run_local_pipeline.py \
    --real-source local \
    --samples 50 \
    --detectors cnndetection,f3net
```

## 📥 第四部分：收割成果与论文素材下载
```python
# 将成果打包下载
!zip -r result_v2_N50_final.zip results/ data/shuffled_8x8/ data/high_pass_residuals/ data/physical_consistency_results.csv

from google.colab import files
files.download('result_v2_N50_final.zip')
```
