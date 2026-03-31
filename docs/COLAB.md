# Google Colab 毕设实验终极执行指南 (Stages 01-09)

本指南针对 AIGC 检测公平性评估全流程（实验组 n=50）。

## ⚙️ 第一部分：环境配置与驱动挂载
建议使用 **L4 GPU** 或 **A100 GPU** 运行。

```python
# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 从 GitHub 拉取毕设框架 (Stage 01-09 完整版)
%cd /content
!rm -rf project 
!git clone https://github.com/wangyh174/biyesheji.git project
%cd /content/project
!git pull origin main

# 3. 安装依赖 (自动包含 CLIP, GLCM, Diffusers 等)
!pip install -r requirements.txt
!pip install -e semantic-image-editing-main/
!pip install opencv-python scikit-image transformers diffusers accelerate
```

## 🛠️ 第二部分：双库真人样本采集
已内置 Pixabay + Pexels 双引擎 Key。
```python
# 采集 50x4 = 200 张经过 CLIP 语义解歧校准的真人对照图
!python scripts/download_real_samples.py --samples-per-group 50
```

## 🚀 第三部分：一键运行总管线 (01-09)
包含生成、偏见计算、Grad-CAM归因、Patch打乱、物理一致性分析及解耦创新方案。
```python
# 开启 50 样本规模的海量评估
!python scripts/00_run_local_pipeline.py \
    --real-source local \
    --samples 50 \
    --detectors cnndetection,f3net
```

## 📥 第四部分：打包归档并下载到本地
运行结束后导出全量论文素材。
```python
# 将成果打包 (热力图、分析表格、对比曲线)
!zip -r result_v2_N50_final.zip results/ data/shuffled_8x8/ data/high_pass_residuals/ data/physical_consistency_results.csv

from google.colab import files
files.download('result_v2_N50_final.zip')
```

---
*祝你的毕设实验顺利跑完并获得完美的公平性分析结果！*
