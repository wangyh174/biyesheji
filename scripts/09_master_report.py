"""
Stage 09: Master Report Generator
Consolidates all experimental results into a single thesis-ready summary.
Aligned with: Fair-Diffusion (CVPR'24), BSA (NeurIPS'24), D3 (ICCV'25), Lin (CVPR'24)
"""

import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Master Report Generator")
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--detectors", type=str, default="cnndetection,lgrad,npr")
    return parser.parse_args()


def load_detector_summary(results_dir: Path, detector: str) -> dict:
    """Load fairness summary JSON for a single detector."""
    json_path = results_dir / "fairness_tables" / detector / "fairness_summary.json"
    if not json_path.exists():
        return {"status": "missing", "detector": detector}
    data = json.loads(json_path.read_text(encoding="utf-8"))
    overall = data.get("overall", {})
    ci = data.get("bootstrap_ci_95", {})
    return {
        "detector": detector,
        "status": "ok",
        "accuracy": overall.get("accuracy", float("nan")),
        "auc": overall.get("auc", float("nan")),
        "max_gap_fpr": overall.get("max_gap_fpr", float("nan")),
        "max_gap_fnr": overall.get("max_gap_fnr", float("nan")),
        "fm_eo_pct": overall.get("fm_eo_pct", float("nan")),
        "fdp_pct": overall.get("fdp_pct", float("nan")),
        "ffpr_pct": overall.get("ffpr_pct", float("nan")),
        "foae_pct": overall.get("foae_pct", float("nan")),
        "ci_fpr_low": ci.get("max_gap_fpr", {}).get("ci_low", float("nan")),
        "ci_fpr_high": ci.get("max_gap_fpr", {}).get("ci_high", float("nan")),
    }


def load_physical_consistency(data_dir: Path) -> pd.DataFrame | None:
    csv_path = data_dir / "physical_consistency_results.csv"
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path)


def compute_auc_from_scores_csv(csv_path: Path) -> float:
    if not csv_path.exists():
        return float("nan")
    df = pd.read_csv(csv_path)
    y = df["y_true"].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, df["score"].astype(float).to_numpy()))


def main() -> None:
    args = parse_args()
    root = args.project_root
    results_dir = root / "results"
    data_dir = root / "data"
    report_dir = results_dir / "master_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    detectors = [d.strip() for d in args.detectors.split(",") if d.strip()]

    # --- 1. Detector Fairness Comparison Table ---
    det_rows = []
    for det in detectors:
        det_rows.append(load_detector_summary(results_dir, det))
    det_df = pd.DataFrame(det_rows)
    det_df.to_csv(report_dir / "detector_fairness_comparison.csv", index=False, encoding="utf-8")
    print("[saved] detector_fairness_comparison.csv")

    # --- 2. Physical Consistency Summary ---
    phys_df = load_physical_consistency(data_dir)
    if phys_df is not None and len(phys_df) > 0:
        phys_summary = phys_df.groupby(["label", "group"]).mean(numeric_only=True).reset_index()
        phys_summary.to_csv(report_dir / "physical_consistency_summary.csv", index=False, encoding="utf-8")
        print("[saved] physical_consistency_summary.csv")
    else:
        print("[warn] No physical consistency data found.")

    # --- 3. Patch Shuffling Impact ---
    shuffled_results = []
    structural_curve_csv = results_dir / "structural_attribution" / "structural_attribution_curve.csv"
    if structural_curve_csv.exists():
        structural_df = pd.read_csv(structural_curve_csv)
        for det in detectors:
            sub = structural_df[structural_df["detector"] == det].copy()
            if len(sub) == 0:
                continue
            baseline = sub[sub["patch_n"] == 1]
            if len(baseline) == 0:
                continue
            baseline_auc = float(baseline["auc"].iloc[0])
            for _, row in sub.sort_values("patch_n").iterrows():
                shuffled_results.append(
                    {
                        "detector": det,
                        "patch_n": int(row["patch_n"]),
                        "auc": float(row["auc"]),
                        "fpr_gap": float(row["fpr_gap"]),
                        "baseline_auc": baseline_auc,
                        "auc_drop": baseline_auc - float(row["auc"]),
                    }
                )
    if shuffled_results:
        shuf_df = pd.DataFrame(shuffled_results)
        shuf_df.to_csv(report_dir / "patch_shuffling_impact.csv", index=False, encoding="utf-8")
        print("[saved] patch_shuffling_impact.csv")

    # --- 4. Master Summary JSON ---
    master = {
        "generated_at": datetime.now().isoformat(),
        "n_detectors": len(detectors),
        "detectors": detectors,
        "detector_results": det_rows,
    }

    # Add high-pass residual analysis if available
    hp_dir = data_dir / "high_pass_residuals"
    if hp_dir.exists():
        hp_results = []
        for det in detectors:
            hp_csv = results_dir / "detector_outputs_highpass" / f"{det}_scores.csv"
            if hp_csv.exists():
                hp_results.append({
                    "detector": det,
                    "highpass_auc": compute_auc_from_scores_csv(hp_csv)
                })
        master["high_pass_results"] = hp_results

    (report_dir / "master_summary.json").write_text(
        json.dumps(master, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8"
    )
    print("[saved] master_summary.json")

    # --- 5. Generate Markdown Report ---
    lines = [
        "# AIGC Detection Fairness - Master Report",
        "",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Detector Fairness Comparison",
        "",
    ]
    if len(det_df) > 0:
        lines.append(det_df.to_markdown(index=False))
    lines += [
        "",
        "## 2. Key Findings",
        "",
        "### FPR Gap Analysis (Lower is Fairer)",
        "",
    ]
    for _, row in det_df.iterrows():
        if row.get("status") == "ok":
            fpr_gap = row.get("max_gap_fpr", float("nan"))
            lines.append(f"- **{row['detector']}**: FPR Gap = {fpr_gap:.4f} "
                        f"(CI: [{row.get('ci_fpr_low', float('nan')):.4f}, "
                        f"{row.get('ci_fpr_high', float('nan')):.4f}])")

    lines += [
        "",
        "## 3. Methodology References",
        "",
        "- **Fair-Diffusion** (Friedrich et al., CVPR 2024): Bias mitigation in text-to-image generation",
        "- **BSA** (Xu et al., NeurIPS 2024): Balanced sensitivity analysis for AI-generated content detection",
        "- **D3** (Zheng et al., ICCV 2025): Second-order physical consistency features (GLCM)",
        "- **Lin et al.** (CVPR 2024): Patch-based structural attribution for deepfake detection",
        "",
    ]

    (report_dir / "MASTER_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("[saved] MASTER_REPORT.md")
    print("\n=== Master Report Generation Complete ===")


if __name__ == "__main__":
    main()
