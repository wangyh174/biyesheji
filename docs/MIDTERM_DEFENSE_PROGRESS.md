# 中期答辩进展说明

## 1. 课题名称

面向医护职业与性别分组的 AIGC 图像检测公平性评估与归因分析

## 2. 研究目标

本课题聚焦于一个具体问题：现有 AI 生成图像检测器在不同性别与职业组合群体上，是否存在系统性的性能差异。为此，本文围绕四个受控分组构建实验对象：

- `male-doctor`
- `female-doctor`
- `male-nurse`
- `female-nurse`

在此基础上，本文进一步分析检测偏差是否主要来源于：

- 图像的全局语义内容
- 生成图像中的局部纹理与高频伪影
- 真实图像与生成图像之间的物理统计差异

## 3. 当前已完成工作

### 3.1 真实图像数据组织

已完成四个目标群体的真实图像收集与目录规范化整理，统一存放于：

- `data/real_samples/male-doctor/`
- `data/real_samples/female-doctor/`
- `data/real_samples/male-nurse/`
- `data/real_samples/female-nurse/`

同时，已实现真实图像自动元数据构建脚本：

- `scripts/build_real_metadata_auto_v2.py`

该脚本当前可完成：

- 按组扫描真实图像
- 去重与可读性检查
- 模糊度与基础质量统计
- 基于 CLIP 的自动审核
- 输出结构化元数据文件，支持后续筛选与实验对齐

### 3.2 生成图像流程搭建

已完成生成阶段主脚本：

- `scripts/01_generate.py`

当前支持三种生成模式：

- `mock`
- `diffusers`
- `fairdiffusion`

其中，中期后确定将主实验生成设定调整为：

- `Fair-Diffusion + Stable Diffusion 1.5`

这一调整的原因是：该设定与参考论文 `Fair-Diffusion` 的实验基础更一致，便于论文表述、实验解释和方法对齐。此前使用的 `Realistic_Vision_V5.1` 更适合作为补充实验，而不宜作为主实验生成器。

### 3.3 生成质量审计与样本筛选

已完成以下两个关键脚本的改造与验证：

- `scripts/01b_generation_audit.py`
- `scripts/02_quality_filter.py`

目前的样本筛选流程已具备：

- CLIP 语义一致性评分
- 目标组别一致性校验
- 组别 margin 约束
- human photo 分数约束
- 与 toy / cartoon / object / deformed face / low-quality 等非目标类型的对抗式比较

同时，`02_quality_filter.py` 已增加中间产物保存能力，可输出：

- `data/metadata_scored.csv`
- `data/metadata_filtered_prebalance.csv`
- `data/metadata_balanced.csv`

这些中间文件用于调试筛选逻辑、分析筛选损失来源，并支撑后续答辩中对实验流程合理性的说明。

### 3.4 检测器推理框架重构

已对检测阶段脚本完成重构：

- `scripts/03_run_detectors.py`

当前版本已不再使用“手工特征 + LogisticRegression”的旧代理实现，而是改为面向公开预训练权重的推理框架，目标检测器包括：

- CNNDetection
- F3Net
- Gram
- LGrad

目前已完成：

- 支持本地权重目录优先加载
- 支持 `--detector all` 顺序运行多个检测器
- F3Net 输出从 logits 改为 softmax 概率，修正阈值语义
- Colab / Drive 目录结构适配
- 兼容手工下载权重后的本地路径解析

在实际运行中，已率先跑通 `CNNDetection` 检测与公平性评估链路。

### 3.5 公平性评估模块完善

已完成公平性评估脚本：

- `scripts/04_fairness_eval.py`

目前支持：

- group 级指标导出
- overall 指标导出
- fairness summary 汇总
- 按 `group x y_true` 分层 bootstrap 置信区间估计

该部分与论文目标直接相关，能够从检测输出中量化不同群体间的 FPR / FNR / accuracy disparity 等关键指标。

### 3.6 可解释性与归因模块完善

已完成或基本完成以下分析模块：

- `scripts/05_gradcam_analysis.py`
- `scripts/06_patch_shuffling_exp.py`
- `scripts/07_physical_consistency.py`
- `scripts/08_high_pass_innovation.py`

其中：

- `05_gradcam_analysis.py` 已改为真实模型反向传播式 Grad-CAM，而非伪热图
- `06_patch_shuffling_exp.py` 用于分析检测器是否过度依赖全局语义结构
- `07_physical_consistency.py` 用于分析真实图像与生成图像在物理统计上的差异
- `08_high_pass_innovation.py` 作为本文创新实验，用于测试检测器在剥离语义信息后是否仍能基于高频残差完成判别

## 4. 当前已经解决的关键技术问题

截至目前，项目中已经明确解决或基本解决的问题包括：

1. 将检测器部分从代理式实验框架升级为预训练权重推理框架，提高实验可信度。
2. 修正了 F3Net 分数解释逻辑，避免将 logits 直接当作概率阈值使用。
3. 完善了公平性 bootstrap 过程，使其与小样本分组场景更匹配。
4. 统一了 Colab + Google Drive 的运行路径与数据持久化方案。
5. 将 `build_real_metadata_auto_v2.py` 和多处脚本改为优先 GPU 运行，并补充进度可视化。
6. 解决了部分外部仓库、外部权重、旧版依赖与新环境之间的兼容性问题。
7. 将主实验生成基座调整为 `Stable Diffusion 1.5`，以保证与参考论文设定一致。

## 5. 当前阶段性结果

目前已形成如下阶段性成果：

- 项目总体实验管线已经搭建完成
- 真实图像分组数据已经基本整理完成
- 生成图像、质量筛选、检测推理、公平性评估链路已经可以运行
- `CNNDetection` 检测器已经能够完成推理与公平性评估
- 代码仓库已形成较完整的脚本化结构，支持后续在 Colab 环境中继续复现实验

当前仓库的核心主线脚本包括：

- `scripts/01_generate.py`
- `scripts/01b_generation_audit.py`
- `scripts/02_quality_filter.py`
- `scripts/03_run_detectors.py`
- `scripts/04_fairness_eval.py`
- `scripts/05_gradcam_analysis.py`
- `scripts/build_real_metadata_auto_v2.py`

## 6. 当前仍在推进的问题

中期阶段后续仍需继续推进的内容主要有：

1. 以 `Fair-Diffusion + Stable Diffusion 1.5` 为主设定，重新生成并整理最终假图样本。
2. 完整跑通剩余检测器的稳定推理流程，尤其是 F3Net / Gram / LGrad 在 Colab 环境中的依赖与权重兼容问题。
3. 进一步稳定 `Grad-CAM`、结构归因、高频残差实验的批量运行流程。
4. 在最终平衡样本规模上，根据真实图像最小组数量确定可用的统一样本数。
5. 汇总正式论文所需的图表、表格与对比结论。

## 7. 下一步计划

接下来的工作安排如下：

### 第一阶段

- 清理旧生成结果
- 以 `Stable Diffusion 1.5` 为主生成基座重跑假图生成
- 重新执行质量筛选并形成最终平衡数据集

### 第二阶段

- 跑通 `CNNDetection` 以外的检测器
- 完成多检测器公平性评估结果汇总
- 生成组间公平性对比表格

### 第三阶段

- 完成 Grad-CAM 可视化
- 完成 patch shuffling 结构归因实验
- 完成高频残差创新实验

### 第四阶段

- 汇总结果
- 撰写论文第四章实验结果与分析
- 完成中后期答辩与最终论文所需材料

## 8. 当前中期结论

截至中期，本文已经完成了课题的核心实验框架搭建，并基本打通了从数据准备、图像生成、质量筛选、检测推理到公平性评估的主线流程。项目目前已经从“方案设计阶段”进入“可重复实验阶段”。后续工作的重点将转向：

- 统一主实验生成设定
- 完整跑通多检测器
- 固化实验结果
- 提炼论文中的公平性结论与归因分析

当前阶段的工作已经能够支撑中期答辩对“研究问题是否明确、技术路线是否可行、阶段成果是否形成”的要求。
