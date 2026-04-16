# AIGC Detection Fairness - Master Report

**Generated**: 2026-04-16 20:26

## 1. Detector Fairness Comparison

| detector     | status   |   accuracy |      auc |   max_gap_fpr |   max_gap_fnr |   fm_eo_pct |   fdp_pct |   ffpr_pct |   foae_pct |   ci_fpr_low |   ci_fpr_high |
|:-------------|:---------|-----------:|---------:|--------------:|--------------:|------------:|----------:|-----------:|-----------:|-------------:|--------------:|
| cnndetection | ok       |   0.652778 | 0.637987 |      0        |      0.285714 |     28.5714 |   11.1111 |     0      |    11.1111 |    0         |      0        |
| lgrad        | ok       |   0.902778 | 0.973214 |      0.272727 |      0.142857 |     27.2727 |   16.6667 |    27.2727 |    16.6667 |    0.0909091 |      0.545455 |
| npr          | ok       |   0.972222 | 0.976461 |      0.181818 |      0        |     18.1818 |   11.1111 |    18.1818 |    11.1111 |    0         |      0.454545 |

## 2. Key Findings

### FPR Gap Analysis (Lower is Fairer)

- **cnndetection**: FPR Gap = 0.0000 (CI: [0.0000, 0.0000])
- **lgrad**: FPR Gap = 0.2727 (CI: [0.0909, 0.5455])
- **npr**: FPR Gap = 0.1818 (CI: [0.0000, 0.4545])

## 3. Methodology References

- **Fair-Diffusion** (Friedrich et al., CVPR 2024): Bias mitigation in text-to-image generation
- **BSA** (Xu et al., NeurIPS 2024): Balanced sensitivity analysis for AI-generated content detection
- **D3** (Zheng et al., ICCV 2025): Second-order physical consistency features (GLCM)
- **Lin et al.** (CVPR 2024): Patch-based structural attribution for deepfake detection
