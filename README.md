# AIGC 检测公平性评估（Colab 优先）

本项目用于评估 `性别 × 职业` 分组下的 AIGC 检测公平性，默认分组：
- `male-doctor`
- `female-doctor`
- `male-nurse`
- `female-nurse`

## 流水线
1. `scripts/01_generate.py`：生成 fake/real 样本（支持 `mock` / `diffusers` / `fairdiffusion`）
2. `scripts/01b_generation_audit.py`：生成侧偏差审计
3. `scripts/02_quality_filter.py`：质量与 CLIP 过滤、组间配平
4. `scripts/03_run_detectors.py`：检测器评估（`cnndetection` / `f3net` / `lgrad`）
5. `scripts/04_fairness_eval.py`：公平性指标与 bootstrap 置信区间
6. `scripts/05_gradcam_analysis.py`：误判样本导出（归因入口）
7. `scripts/06_consolidate_results.py`：统一汇总当前结果

## 本地快速测试
```bash
python scripts/00_run_local_pipeline.py \
  --project-root . \
  --generator mock \
  --samples-per-group 30 \
  --real-per-group 30 \
  --detectors cnndetection,f3net,lgrad \
  --clip-min-score 0.10
```

## Colab 运行
请按 [docs/COLAB.md](D:\desktop\bishe\project\docs\COLAB.md) 执行（包含完整可复制单元）。

## 关键输出
- `results/generation_audit/generation_audit.json`
- `results/fairness_tables/<detector>/overall_metrics.csv`
- `results/fairness_tables/<detector>/fairness_summary.json`
- `results/fairness_tables/latest_run_overview.csv`
- `results/attribution/misclassified_samples.csv`

## 说明
- 已清理历史运行残留（`results`、`data/generated_raw`、`data/real_samples`、运行型 metadata 已重置）。
- 当前项目不要求下载 Stable Diffusion 全仓文件；按 Colab 文档中的“核心文件/在线拉取”方式即可。
