# Colab 运行指南（毕业设计最终完整版）

这份文档记录了如何在 Google Colab 无痛跑通整个包含“受控图像生成”、“公平性检测打分”、“热力图可视归因”的毕业设计流水线，并自动防止云端断联导致生图数据丢失。

## 1. 新建 Colab 并开启 GPU
- 打开一段新的笔记本
- 菜单：`Runtime (代码执行程序)` -> `Change runtime type (更改运行时类型)` -> `Hardware accelerator (硬件加速器)` 选择 `T4 GPU`。

## 2. 终极一键全自动运行代码块
这个代码块极其智能，**你只需要建这一个代码块（Code cell）**：它会自动挂载你的 Google 硬盘、拉取最新代码、安装防依赖冲突的环境、并且开启高画质大模型（Realistic Vision）生图，同时自动打分并抓取错判样本画出热力图。

不管它运行多久，由于工作目录已经物理绑定到你的硬盘里运行，如果网页中途被关闭，你已经生成的心血图片**绝对不会被清空丢失！** 下次进来重新点一次这个代码块就能无缝接着跑。

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
    print("云端硬盘里还没有代码，正在克隆你 Github 上的无敌版代码...")
    !git clone https://github.com/wangyh174/biyesheji.git project
else:
    print("代码库已存在，进入工作状态...")
    %cd {project_dir}
    !git pull origin main

%cd {project_dir}

# 4. 一键安装依赖防打架版（Pillow / Diffusers 等库锁定）
!pip -q install -r requirements-colab.txt
!pip -q install -e ./semantic-image-editing-main/semantic-image-editing-main

# 5. 起飞！启动正式带反向提示词、强迫真实图像对照、和全自动热力归因的全量分析！
!python scripts/00_run_local_pipeline.py \
  --project-root ./ \
  --generator fairdiffusion \
  --samples-per-group 40 \
  --real-per-group 40 \
  --detectors cnndetection,f3net,lgrad \
  --clip-min-score 0.22 \
  --real-source diffusers \
  --model-id SG161222/Realistic_Vision_V5.1_noVAE \
  --real-model-id SG161222/Realistic_Vision_V5.1_noVAE
```

## 3. 关键成果查收
等待上面长长的生图加载条全部跑满 100% 并且各个检测器打分完毕后，毕设原始数据会自动塞在你的 Google 硬盘里：
- **最终各模型打分与偏差表格：** `/MyDrive/Bishe/project/results/fairness_tables/`
- **生成的大量逼真假图与真图样本：** `/MyDrive/Bishe/project/data/`
- **误判的热力特征分析图（可直接插论文）：** `/MyDrive/Bishe/project/results/attribution/heatmaps/`
