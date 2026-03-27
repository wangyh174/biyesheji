"""
Step 4: Fairness evaluation (Accuracy/FPR/FNR disparities) + bootstrap CI.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detector-csv",
        type=Path,
        default=root / "results" / "detector_outputs" / "baseline_scores.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "fairness_tables")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "all"])
    parser.add_argument("--bootstrap-iters", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b != 0 else float("nan")


def group_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    for g, sub in df.groupby("group"):
        y = sub["y_true"].astype(int).to_numpy()
        y_hat = sub["y_hat"].astype(int).to_numpy()
        score = sub["score"].astype(float).to_numpy()

        tp = int(np.sum((y == 1) & (y_hat == 1)))
        tn = int(np.sum((y == 0) & (y_hat == 0)))
        fp = int(np.sum((y == 0) & (y_hat == 1)))
        fn = int(np.sum((y == 1) & (y_hat == 0)))

        fpr = safe_div(fp, fp + tn)
        fnr = safe_div(fn, fn + tp)
        err = safe_div(fp + fn, len(sub))
        acc = safe_div(tp + tn, len(sub))

        auc = float("nan")
        if len(np.unique(y)) == 2:
            auc = float(roc_auc_score(y, score))

        rows.append(
            {
                "group": g,
                "n": int(len(sub)),
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "fpr": fpr,
                "fnr": fnr,
                "error_rate": err,
                "accuracy": acc,
                "auc": auc,
            }
        )
    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)


def fairness_summary(gdf: pd.DataFrame) -> Dict[str, float]:
    acc_vals = gdf["accuracy"].dropna().to_numpy()
    fpr_vals = gdf["fpr"].dropna().to_numpy()
    fnr_vals = gdf["fnr"].dropna().to_numpy()
    err_vals = gdf["error_rate"].dropna().to_numpy()
    return {
        "accuracy_disparity": float(np.max(acc_vals) - np.min(acc_vals)) if len(acc_vals) > 0 else float("nan"),
        "max_gap_fpr": float(np.max(fpr_vals) - np.min(fpr_vals)) if len(fpr_vals) > 0 else float("nan"),
        "max_gap_fnr": float(np.max(fnr_vals) - np.min(fnr_vals)) if len(fnr_vals) > 0 else float("nan"),
        "worst_group_error": float(np.max(err_vals)) if len(err_vals) > 0 else float("nan"),
    }


def _pairwise_max_abs(vals: List[float]) -> float:
    arr = np.asarray([v for v in vals if not np.isnan(v)], dtype=np.float64)
    if len(arr) == 0:
        return float("nan")
    if len(arr) == 1:
        return 0.0
    return float(np.max(arr) - np.min(arr))


def cvpr2024_style_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    CVPR 2024 fairness-style metrics (lower is better), reported in [0,1]:
    - FDP  : max demographic parity gap over groups
    - FM-EO: max equalized-odds gap over y in {0,1}
    - FFPR : max false-positive-rate gap over groups
    - FOAE : max overall-accuracy gap over groups
    """
    groups = sorted(df["group"].astype(str).unique().tolist())

    # FDP: max_g,g' | P(y_hat=1 | g) - P(y_hat=1 | g') |
    dp_vals = []
    for g in groups:
        sub = df[df["group"] == g]
        dp_vals.append(float(np.mean(sub["y_hat"].astype(int).to_numpy() == 1)))
    fdp = _pairwise_max_abs(dp_vals)

    # FM-EO: max over y∈{0,1} of pairwise gap in P(y_hat=1 | g, y)
    eo_gaps = []
    for yv in [0, 1]:
        per_group = []
        for g in groups:
            sub = df[(df["group"] == g) & (df["y_true"].astype(int) == yv)]
            if len(sub) == 0:
                per_group.append(float("nan"))
            else:
                per_group.append(float(np.mean(sub["y_hat"].astype(int).to_numpy() == 1)))
        eo_gaps.append(_pairwise_max_abs(per_group))
    fm_eo = float(np.nanmax(np.asarray(eo_gaps, dtype=np.float64)))

    # FFPR: max_g,g' | FPR_g - FPR_g' |
    fpr_vals = []
    for g in groups:
        sub = df[df["group"] == g]
        y = sub["y_true"].astype(int).to_numpy()
        y_hat = sub["y_hat"].astype(int).to_numpy()
        tn = float(np.sum((y == 0) & (y_hat == 0)))
        fp = float(np.sum((y == 0) & (y_hat == 1)))
        fpr_vals.append(safe_div(fp, fp + tn))
    ffpr = _pairwise_max_abs(fpr_vals)

    # FOAE: max_g,g' | Acc_g - Acc_g' |
    acc_vals = []
    for g in groups:
        sub = df[df["group"] == g]
        y = sub["y_true"].astype(int).to_numpy()
        y_hat = sub["y_hat"].astype(int).to_numpy()
        acc_vals.append(float(np.mean(y == y_hat)))
    foae = _pairwise_max_abs(acc_vals)

    return {
        "fm_eo": fm_eo,
        "fdp": fdp,
        "ffpr": ffpr,
        "foae": foae,
        "fm_eo_pct": fm_eo * 100.0,
        "fdp_pct": fdp * 100.0,
        "ffpr_pct": ffpr * 100.0,
        "foae_pct": foae * 100.0,
    }


def bootstrap_ci(df: pd.DataFrame, iters: int, seed: int) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    vals = {
        "accuracy_disparity": [],
        "max_gap_fpr": [],
        "max_gap_fnr": [],
        "worst_group_error": [],
        "fm_eo": [],
        "fdp": [],
        "ffpr": [],
        "foae": [],
    }

    for _ in range(iters):
        sample_idx = rng.integers(0, len(df), size=len(df))
        boot = df.iloc[sample_idx].reset_index(drop=True)
        gdf = group_metrics(boot)
        summ = fairness_summary(gdf)
        cvpr_m = cvpr2024_style_metrics(boot)
        for k in vals:
            v = summ[k] if k in summ else cvpr_m[k]
            if not np.isnan(v):
                vals[k].append(v)

    out: Dict[str, Dict[str, float]] = {}
    for k, arr in vals.items():
        if len(arr) == 0:
            out[k] = {"ci_low": float("nan"), "ci_high": float("nan")}
        else:
            out[k] = {
                "ci_low": float(np.percentile(arr, 2.5)),
                "ci_high": float(np.percentile(arr, 97.5)),
            }
    return out


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.detector_csv)
    if args.split != "all":
        df = df[df["split"] == args.split].copy().reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("No data after split filtering.")

    gdf = group_metrics(df)
    summ = fairness_summary(gdf)

    y_true = df["y_true"].astype(int).to_numpy()
    score = df["score"].astype(float).to_numpy()
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        auc = float(roc_auc_score(y_true, score))

    overall = {
        "n": int(len(df)),
        "detector_name": str(df["detector_name"].iloc[0]) if "detector_name" in df.columns else "unknown",
        "accuracy": float(np.mean(y_true == df["y_hat"].astype(int).to_numpy())),
        "auc": auc,
        **summ,
    }
    cvpr_m = cvpr2024_style_metrics(df)
    overall.update(cvpr_m)
    ci = bootstrap_ci(df, iters=args.bootstrap_iters, seed=args.seed)

    group_path = args.output_dir / "group_metrics.csv"
    overall_path = args.output_dir / "overall_metrics.csv"
    summary_json = args.output_dir / "fairness_summary.json"

    gdf.to_csv(group_path, index=False, encoding="utf-8")
    pd.DataFrame([overall]).to_csv(overall_path, index=False, encoding="utf-8")

    payload = {"overall": overall, "bootstrap_ci_95": ci}
    summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[saved] group metrics: {group_path}")
    print(f"[saved] overall metrics: {overall_path}")
    print(f"[saved] fairness summary: {summary_json}")
    print(
        "[metric] "
        f"acc_disp={overall['accuracy_disparity']:.4f} "
        f"max_gap_fpr={overall['max_gap_fpr']:.4f} "
        f"max_gap_fnr={overall['max_gap_fnr']:.4f} "
        f"worst_group_error={overall['worst_group_error']:.4f} "
        f"FM-EO={overall['fm_eo_pct']:.2f}% "
        f"FDP={overall['fdp_pct']:.2f}% "
        f"FFPR={overall['ffpr_pct']:.2f}% "
        f"FOAE={overall['foae_pct']:.2f}%"
    )


if __name__ == "__main__":
    main()
