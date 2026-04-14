from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


VALID_GROUPS = [
    "male-doctor",
    "female-doctor",
    "male-nurse",
    "female-nurse",
]

FACE_CATS = ["close", "chest", "half", "full"]
SCENE_CATS = [
    "hospital_corridor",
    "clinic_room",
    "ward",
    "nursing_station",
    "plain_bg",
    "office_like",
    "other",
]
CLOTHING_CATS = ["white_coat", "scrubs", "mixed", "unclear"]
ITEM_CATS = ["stethoscope", "mask", "badge", "none"]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Summarize group-level distribution checks for real image metadata."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=root / "data" / "real_metadata_auto.csv",
        help="Input metadata CSV produced by build_real_metadata_auto.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "results" / "real_group_audit",
        help="Directory to save summary tables.",
    )
    parser.add_argument(
        "--only-kept",
        action="store_true",
        help="If set, only use rows with manual_keep == 1.",
    )
    return parser.parse_args()


def percent_table(df: pd.DataFrame, col: str, categories: List[str]) -> pd.DataFrame:
    rows = []
    for group in VALID_GROUPS:
        sub = df[df["group"] == group].copy()
        total = len(sub)
        row: Dict[str, object] = {"group": group, "total": int(total)}
        for cat in categories:
            count = int((sub[col].astype(str) == cat).sum()) if total > 0 else 0
            pct = (count / total * 100.0) if total > 0 else 0.0
            row[f"{cat}_count"] = count
            row[f"{cat}_pct"] = round(pct, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in VALID_GROUPS:
        sub = df[df["group"] == group].copy()
        if len(sub) == 0:
            rows.append(
                {
                    "group": group,
                    "total": 0,
                    "width_mean": None,
                    "height_mean": None,
                    "num_pixels_mean": None,
                    "blur_score_mean": None,
                    "review_flag_rate_pct": None,
                    "duplicate_count": 0,
                }
            )
            continue

        duplicate_count = int(sub["duplicate_hash"].value_counts().gt(1).sum())
        rows.append(
            {
                "group": group,
                "total": int(len(sub)),
                "width_mean": round(float(pd.to_numeric(sub["width"], errors="coerce").mean()), 2),
                "height_mean": round(float(pd.to_numeric(sub["height"], errors="coerce").mean()), 2),
                "num_pixels_mean": round(float(pd.to_numeric(sub["num_pixels"], errors="coerce").mean()), 2),
                "blur_score_mean": round(float(pd.to_numeric(sub["blur_score"], errors="coerce").mean()), 2),
                "review_flag_rate_pct": round(float((pd.to_numeric(sub["review_flag"], errors="coerce") == 1).mean() * 100.0), 2),
                "duplicate_count": duplicate_count,
            }
        )
    return pd.DataFrame(rows)


def comparison_table(df: pd.DataFrame, col: str, categories: List[str]) -> pd.DataFrame:
    records = []
    pct_df = percent_table(df, col, categories)
    for cat in categories:
        pct_col = f"{cat}_pct"
        vals = pct_df[pct_col].fillna(0.0).tolist()
        records.append(
            {
                "attribute": col,
                "category": cat,
                "max_pct": round(max(vals), 2) if vals else 0.0,
                "min_pct": round(min(vals), 2) if vals else 0.0,
                "max_gap_pct": round((max(vals) - min(vals)), 2) if vals else 0.0,
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if args.only_kept:
        df = df[df["manual_keep"].astype(str) == "1"].copy().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No rows available after filtering.")

    overall_summary = numeric_summary(df)
    face_summary = percent_table(df, "face_scale_auto", FACE_CATS)
    scene_summary = percent_table(df, "scene_type_auto", SCENE_CATS)
    clothing_summary = percent_table(df, "clothing_type_auto", CLOTHING_CATS)
    item_summary = percent_table(df, "medical_item_auto", ITEM_CATS)

    face_gap = comparison_table(df, "face_scale_auto", FACE_CATS)
    scene_gap = comparison_table(df, "scene_type_auto", SCENE_CATS)
    clothing_gap = comparison_table(df, "clothing_type_auto", CLOTHING_CATS)
    item_gap = comparison_table(df, "medical_item_auto", ITEM_CATS)
    gap_summary = pd.concat([face_gap, scene_gap, clothing_gap, item_gap], ignore_index=True)

    overall_summary.to_csv(args.output_dir / "overall_numeric_summary.csv", index=False, encoding="utf-8")
    face_summary.to_csv(args.output_dir / "face_scale_distribution.csv", index=False, encoding="utf-8")
    scene_summary.to_csv(args.output_dir / "scene_distribution.csv", index=False, encoding="utf-8")
    clothing_summary.to_csv(args.output_dir / "clothing_distribution.csv", index=False, encoding="utf-8")
    item_summary.to_csv(args.output_dir / "medical_item_distribution.csv", index=False, encoding="utf-8")
    gap_summary.to_csv(args.output_dir / "distribution_gap_summary.csv", index=False, encoding="utf-8")

    print(f"[saved] overall numeric summary: {args.output_dir / 'overall_numeric_summary.csv'}")
    print(f"[saved] face scale summary: {args.output_dir / 'face_scale_distribution.csv'}")
    print(f"[saved] scene summary: {args.output_dir / 'scene_distribution.csv'}")
    print(f"[saved] clothing summary: {args.output_dir / 'clothing_distribution.csv'}")
    print(f"[saved] medical item summary: {args.output_dir / 'medical_item_distribution.csv'}")
    print(f"[saved] gap summary: {args.output_dir / 'distribution_gap_summary.csv'}")


if __name__ == "__main__":
    main()
