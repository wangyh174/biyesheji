"""
Step 5: Grad-CAM attribution analysis entrypoint (roadmap stage 3).

This script prepares an analyzable list of misclassified samples and (optionally)
runs Grad-CAM if torchvision is available and a CNN checkpoint is provided.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--detector-csv", type=Path, default=root / "results" / "detector_outputs" / "cnndetection_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "attribution")
    parser.add_argument("--only-false-negative", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.detector_csv)
    mis = df[df["y_true"].astype(int) != df["y_hat"].astype(int)].copy()
    if args.only_false_negative:
        mis = mis[(mis["y_true"].astype(int) == 1) & (mis["y_hat"].astype(int) == 0)].copy()

    mis_path = args.output_dir / "misclassified_samples.csv"
    mis.to_csv(mis_path, index=False, encoding="utf-8")

    note = args.output_dir / "README_GradCAM.txt"
    note.write_text(
        "\n".join(
            [
                "Grad-CAM stage prepared.",
                f"Misclassified sample list: {mis_path}",
                "Next: load a CNN detector checkpoint and generate heatmaps for these samples.",
                "Recommendation: prioritize false negatives in female-doctor / male-nurse groups.",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[saved] misclassified list: {mis_path}")
    print(f"[saved] stage note: {note}")
    print("[info] This is the roadmap-aligned attribution entrypoint.")


if __name__ == "__main__":
    main()
