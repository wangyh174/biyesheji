# AIGC 公平性评估与归因分析框架 (Fair-AIGC-Eval)

本项目提供了一套面向特定人口统计学场景（医护群体）的 AIGC 检测器公平性评估与机理分析框架。基于 Fair-Diffusion 受控生成、CLIP 语义对齐、及多层级归因分析（热力图、Patch Shuffle、二阶统计），深度揭示检测器的语义偏见来源。

## 🔬 研究路线图 (Stages 01-09)

本项目严格遵循 **Lin (CVPR 24)**、**BSA (NeurIPS 24)** 及 **D3 (ICCV 25)** 的学术规范：

1.  **Stage 01: 受控生成 (Fair-Diffusion)** - 利用编辑向量（Editing Direction）消除生成侧的原始偏见，生成男女对等的医护样本。
2.  **Stage 02: 质量对齐 (CLIP Filter)** - 使用 CLIP 模型进行语义与质量双向过滤，对齐 Fake/Real 样本池。
3.  **Stage 03 & 04: 基准评估 (Baseline)** - 运行 CNNDetection, F3Net, Gram, LGrad 的预训练权重推理并计算 FPR Gap 和 Fairness Metrics。
4.  **Stage 05: 视觉归因 (Visual Attribution)** - 绘制 Grad-CAM 热力图，定位语义过拟合（Semantic Overfitting）。
5.  **Stage 06: 结构性验证 (Patch Shuffling)** - 在物理块乱序条件下测试偏见趋势，验证偏见是否源于宏观语义。
6.  **Stage 07: 物理本质验证 (Second-Order Stats)** - 基于 GLCM（灰度共生矩阵）验证 AI 生成在二阶特征上的物理一致性。
7.  **Stage 08: 创新方案 (High-pass Decoupling)** - 引入高频残差滤波器实现“语义解耦检测”，提升跨人群公平性。
8.  **Stage 09: 汇总报告 (Master Report)** - 自动化产生全量实验表格、对比柱状图及偏见演化曲线。

## 📊 核心理论参考

-   **Fair-Diffusion (2023)**: 隐空间语义干预框架。
-   **Lin et al. (CVPR 2024)**: 公平性指标定义及局部性增强。
-   **Breaking Semantic Artifacts (NeurIPS 2024)**: Patch Shuffle 与语义伪像理论。
-   **Zheng-D3 (ICCV 2025)**: 基于二阶特征的无监督检测思路。

## 🚀 快速启动

执行以下单条命令即可运行完整毕设流程：
```bash
python scripts/00_run_local_pipeline.py --real-source local --detectors cnndetection,f3net,gram,lgrad
```

数据与结果将存储在 `data/` 和 `results/` 目录下。
观察 `data/shuffled_8x8` 和 `data/high_pass_residuals` 的检测结果以获取深度归因结论。

补充说明：
- `scripts/03_run_detectors.py` 已切换为预训练检测器推理入口，不再使用旧版代理特征分类器。
- 首次运行检测阶段时，会自动下载公开源码压缩包与权重到 `.external_models/`。
