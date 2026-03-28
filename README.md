# AIGC 检测公平性评估（毕业设计）

面向 `性别 × 职业` 分组场景，评估 AIGC 检测器在不同群体上的性能差异，输出可用于论文的公平性指标与审计结果。

## 项目简介
- 研究对象：AIGC 生成内容检测中的组间公平性
- 分组设置：`male-doctor`、`female-doctor`、`male-nurse`、`female-nurse`
- 核心能力：
  - 生成侧偏差审计（数量/质量/来源）
  - 质量过滤与组间配平
  - 多检测器对比（`cnndetection` / `f3net` / `lgrad`）
  - 公平性指标评估（含 CVPR 风格指标）
  - 误判样本归因入口（Grad-CAM）

## 流程图
```mermaid
flowchart LR
    A["01_generate.py<br/>生成 fake/real 样本"] --> B["01b_generation_audit.py<br/>生成侧偏差审计"]
    B --> C["02_quality_filter.py<br/>质量过滤 + CLIP + 配平"]
    C --> D["03_run_detectors.py<br/>运行多检测器"]
    D --> E["04_fairness_eval.py<br/>公平性指标 + CI"]
    E --> F["05_gradcam_analysis.py<br/>误判样本导出"]
    E --> G["06_consolidate_results.py<br/>汇总最新结果"]
```

## 快速开始

### 1) 本地快速验证（mock）
```bash
python scripts/00_run_local_pipeline.py \
  --project-root . \
  --generator mock \
  --samples-per-group 30 \
  --real-per-group 30 \
  --detectors cnndetection,f3net,lgrad \
  --clip-min-score 0.10
```

### 2) 正式实验（fairdiffusion）
```bash
python scripts/00_run_local_pipeline.py \
  --project-root . \
  --generator fairdiffusion \
  --samples-per-group 20 \
  --real-per-group 20 \
  --detectors cnndetection,f3net,lgrad \
  --clip-min-score 0.10
```

### 3) Colab 运行
详见：[docs/COLAB.md](docs/COLAB.md)

## 结果示例
以下为一次 mock 实验输出的示意（以 `latest_run_overview.csv` 为准）：

| detector | accuracy | FM-EO(%) | FDP(%) | FFPR(%) | FOAE(%) |
|---|---:|---:|---:|---:|---:|
| cnndetection | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| f3net | 1.000 | 0.00 | 0.00 | 0.00 | 0.00 |
| lgrad | 1.000 | 22.22 | 11.11 | 0.00 | 11.11 |

## 关键输出
- `results/generation_audit/generation_audit.json`
- `results/fairness_tables/<detector>/overall_metrics.csv`
- `results/fairness_tables/<detector>/fairness_summary.json`
- `results/fairness_tables/latest_run_overview.csv`
- `results/fairness_tables/latest_run_notes.md`
- `results/attribution/misclassified_samples.csv`

## 目录结构（核心）
```text
scripts/
  00_run_local_pipeline.py
  01_generate.py
  01b_generation_audit.py
  02_quality_filter.py
  03_run_detectors.py
  04_fairness_eval.py
  05_gradcam_analysis.py
  06_consolidate_results.py
docs/
  COLAB.md
results/
data/
```

## 备注
- `paper/` 目录默认本地使用，已在 `.gitignore` 中忽略，不会推送。
- 若需要复现实验，请优先固定随机种子与参数配置。
