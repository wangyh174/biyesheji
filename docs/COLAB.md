# Google Colab 毕设实验终极执行指南 (全量审计修复版)

本指南针对 AIGC 检测公平性评估全流程。
对标文献: Fair-Diffusion (CVPR'24), BSA (NeurIPS'24), D3 (ICCV'25), Lin (CVPR'24)

## ⚙️ 第一部分：环境配置与驱动挂载
建议使用 **L4 GPU** 或 **A100 GPU** 运行。

```python
# 1. 挂载 Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. 拉取毕设框架 (已全量修复版)
%cd /content
!rm -rf project 
!git clone https://github.com/wangyh174/biyesheji.git project
%cd /content/project
!git pull origin main

# 3. 安装依赖
!pip install -r requirements.txt
!pip install -e semantic-image-editing-main/semantic-image-editing-main/
!pip install opencv-python scikit-image transformers diffusers accelerate tabulate gdown
```

说明：
- `scripts/03_run_detectors.py` 已切换为预训练权重推理，不再使用旧版的手工特征 + LogisticRegression 代理分类器。
- `CNNDetection / Gram / LGrad` 会在首次运行时自动下载公开检测框架与对应 checkpoint。
- `F3Net` 会在首次运行时自动下载公开发布的 F3Net checkpoint 与依赖源码。
- `scripts/04_fairness_eval.py` 已使用按 `group × y_true` 分层的 bootstrap 估计公平性置信区间。
- `scripts/05_gradcam_analysis.py` 已切换为真实模型反向传播的 Grad-CAM，可直接生成论文级热力图。
- 首次跑 Stage 03 会比以前慢，这是正常现象。

## 🛠️ 第二部分：双库精选真人样本采集 (n=60 候选 → Top 50 精选)
利用"语义解歧"机制，过滤掉"像医生的护士"。超额采集 60 张，后续由 Stage 02 择优保留 50。
```python
# 采集 240 张 (60×4) 经过 CLIP 语义解歧校准的真人对照图
# 已内置 Pexels + Pixabay 双 Key，无需填写
!python scripts/download_real_samples.py \
    --samples-per-group 60 \
    --clip-threshold 0.22
```

## 🚀 第三部分：执行全量毕设管线 (01-09)
包含生成、基准评价、视觉/结构/物理归因及解耦创新。
4 款检测器 (CNNDetection, F3Net, Gram, LGrad) × 30 步推理 × 512 分辨率。
```python
# 开启自动化深度评估 (生成 60 张 → CLIP 精选 Top 50)
%cd /content/project
!python scripts/00_run_local_pipeline.py \
    --real-source local \
    --samples 50 \
    --detectors cnndetection,f3net,gram,lgrad
```

如果你想先单独检查某一个检测器是否下载成功，可以先跑：

```python
%cd /content/project
!python scripts/03_run_detectors.py \
    --detector cnndetection \
    --metadata-in data/metadata_balanced.csv
```

如果你想单独测试某个检测器的 Grad-CAM 是否正常，可在跑完 Stage 03 后执行：

```python
%cd /content/project
!python scripts/05_gradcam_analysis.py \
    --detector-csv results/detector_outputs/cnndetection_scores.csv \
    --analyze-all \
    --max-per-group 2
```

四个检测器建议按下面的理解写入论文或答辩：
- `CNNDetection`: Wang et al. 的经典 CNN 生成图检测器。
- `F3Net`: 频域线索驱动的伪造检测器。
- `Gram`: GramNet，强调纹理/统计关系建模。
- `LGrad`: 基于梯度伪影表示的检测器，强调泛化伪影而非原图语义。

## 📥 第四部分：收割成果 (本地下载 + Drive 永存备份)
```python
# 1. 打包所有核心实验数据
%cd /content/project
!zip -r result_v2_N50_final.zip \
    results/ \
    data/shuffled_8x8/ \
    data/high_pass_residuals/ \
    data/physical_consistency_results.csv

# 2. 自动下载到本地 (浏览器会弹出下载框)
from google.colab import files
try:
    files.download('result_v2_N50_final.zip')
except:
    print("浏览器下载请求已发出 (若被拦截，请手动在左侧文件栏右键下载)")

# 3. 同步至 Google Drive 永久存储
import os
backup_dir = "/content/drive/MyDrive/bishe_project_results/"
!mkdir -p {backup_dir}
print(f"正在持久化备份至 Drive: {backup_dir}")

!cp result_v2_N50_final.zip {backup_dir}
!cp -r results/ {backup_dir}
print("备份完成！你可以在 Google Drive 的 bishe_project_results 文件夹中查看。")
```
