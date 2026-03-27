# Colab 运行指南

## 1. 新建 Colab 并开启 GPU
- Runtime -> Change runtime type -> Hardware accelerator: `GPU`

## 2. 拉取代码
如果你有 GitHub 仓库：
```python
!git clone <你的仓库地址> /content/project
%cd /content/project
```

如果你是本地打包上传：
```python
from google.colab import files
uploaded = files.upload()  # 上传 project.zip
!unzip -q project.zip -d /content
%cd /content/project
```

## 3. 安装依赖
```python
!pip -q install -r requirements-colab.txt
!pip -q install -e ./semantic-image-editing-main/semantic-image-editing-main
```

## 4. 运行（mock 快速）
```python
!python scripts/00_run_local_pipeline.py \
  --project-root /content/project \
  --generator mock \
  --samples-per-group 30 \
  --real-per-group 30 \
  --detectors cnndetection,f3net,lgrad \
  --clip-min-score 0.10
```

## 5. 运行（fairdiffusion）
```python
!python scripts/00_run_local_pipeline.py \
  --project-root /content/project \
  --generator fairdiffusion \
  --samples-per-group 20 \
  --real-per-group 20 \
  --detectors cnndetection,f3net,lgrad \
  --clip-min-score 0.10
```

## 6. 导出结果到 Drive（可选）
```python
from google.colab import drive
drive.mount('/content/drive')
!cp -r /content/project/results /content/drive/MyDrive/bishe_results
```

## 7. 结果查看
- `/content/project/results/fairness_tables/latest_run_overview.csv`
- `/content/project/results/fairness_tables/<detector>/fairness_summary.json`
