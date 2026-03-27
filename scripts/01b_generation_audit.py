"""
Stage 1.5: Generation-side bias audit (Fair-Diffusion angle).

Purpose:
- Separate generation-side distribution issues from detector-side bias.
- Output audit report before detector evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from PIL import Image


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--metadata-in", type=Path, default=root / "data" / "metadata_raw.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "generation_audit")
    parser.add_argument("--save-controlled-metadata", action="store_true")
    parser.add_argument("--controlled-metadata-out", type=Path, default=root / "data" / "metadata_gen_control.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def image_quality_score(path: str) -> float:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    gy, gx = np.gradient(arr)
    grad_energy = float(np.mean(gx * gx + gy * gy))
    contrast = float(np.std(arr))
    hist, _ = np.histogram(arr, bins=64, range=(0.0, 1.0), density=True)
    hist = hist + 1e-12
    entropy = float(-np.sum(hist * np.log(hist)))
    return 0.5 * grad_energy + 0.3 * contrast + 0.2 * entropy


def max_gap(vals: List[float]) -> float:
    arr = np.asarray(vals, dtype=np.float64)
    return float(np.max(arr) - np.min(arr)) if len(arr) > 0 else float("nan")


def controlled_resample(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = []
    for y in sorted(df["y_true"].astype(int).unique()):
        sub = df[df["y_true"].astype(int) == y].copy()
        counts = sub.groupby("group")["id"].count()
        min_n = int(counts.min())

        # Keep center-quality band per group to reduce generation-side quality bias.
        for g in sorted(sub["group"].unique()):
            gs = sub[sub["group"] == g].copy()
            q10 = gs["quality_score"].quantile(0.10)
            q90 = gs["quality_score"].quantile(0.90)
            band = gs[(gs["quality_score"] >= q10) & (gs["quality_score"] <= q90)]
            if len(band) < min_n:
                band = gs
            sampled = band.sample(n=min_n, random_state=seed, replace=False)
            out.append(sampled)
    return pd.concat(out, ignore_index=True)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata_in)
    if len(df) == 0:
        raise ValueError(f"Empty metadata: {args.metadata_in}")

    if "quality_score" not in df.columns:
        df["quality_score"] = np.nan
    df["quality_score"] = [image_quality_score(str(p)) for p in df["file_path"].tolist()]

    group_rows = []
    for (y, g), sub in df.groupby([df["y_true"].astype(int), "group"]):
        group_rows.append(
            {
                "y_true": int(y),
                "group": g,
                "n": int(len(sub)),
                "quality_mean": float(sub["quality_score"].mean()),
                "quality_std": float(sub["quality_score"].std(ddof=0)),
                "source_model": str(sub["source_model"].mode().iloc[0]) if "source_model" in sub.columns else "unknown",
            }
        )
    group_df = pd.DataFrame(group_rows).sort_values(["y_true", "group"]).reset_index(drop=True)
    group_df.to_csv(args.output_dir / "generation_group_stats.csv", index=False, encoding="utf-8")

    summary: Dict[str, float] = {}
    for y in sorted(df["y_true"].astype(int).unique()):
        sub = group_df[group_df["y_true"] == y]
        summary[f"count_gap_y{y}"] = max_gap(sub["n"].astype(float).tolist())
        summary[f"quality_gap_y{y}"] = max_gap(sub["quality_mean"].astype(float).tolist())

    source_counts = (
        df.groupby(["y_true", "source_model"])["id"].count().reset_index().rename(columns={"id": "n"})
    )
    source_counts.to_csv(args.output_dir / "generation_source_counts.csv", index=False, encoding="utf-8")

    payload = {
        "summary": summary,
        "notes": [
            "This audit captures generation-side imbalance before detector training/evaluation.",
            "If count/quality gaps are large, detector fairness conclusions may be confounded.",
        ],
    }
    (args.output_dir / "generation_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.save_controlled_metadata:
        controlled = controlled_resample(df, seed=args.seed)
        args.controlled_metadata_out.parent.mkdir(parents=True, exist_ok=True)
        controlled.to_csv(args.controlled_metadata_out, index=False, encoding="utf-8")
        print(f"[saved] controlled metadata: {args.controlled_metadata_out}")

    print(f"[saved] generation audit: {args.output_dir / 'generation_audit.json'}")
    print(f"[saved] generation group stats: {args.output_dir / 'generation_group_stats.csv'}")
    print(f"[saved] generation source counts: {args.output_dir / 'generation_source_counts.csv'}")


if __name__ == "__main__":
    main()
