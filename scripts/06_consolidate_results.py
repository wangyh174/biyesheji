"""
Stage 6: Consolidate detector/fairness outputs into a single latest-run overview.

Goals:
- Provide one table for thesis writing.
- Mark stale artifacts that are not aligned with current metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--metadata", type=Path, default=root / "data" / "metadata_balanced.csv")
    parser.add_argument("--detectors", type=str, default="cnndetection,f3net,gram,lgrad")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "results" / "fairness_tables" / "latest_run_overview.csv",
    )
    parser.add_argument(
        "--output-notes",
        type=Path,
        default=root / "results" / "fairness_tables" / "latest_run_notes.md",
    )
    return parser.parse_args()


def safe_float(payload: Dict[str, object], key: str) -> float:
    v = payload.get(key, float("nan"))
    try:
        return float(v)
    except Exception:
        return float("nan")


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(args.metadata)
    metadata_rows = int(len(meta))
    metadata_groups = int(meta["group"].nunique()) if "group" in meta.columns else 0

    detectors = [x.strip() for x in args.detectors.split(",") if x.strip()]
    rows: List[Dict[str, object]] = []

    for det in detectors:
        score_csv = args.project_root / "results" / "detector_outputs" / f"{det}_scores.csv"
        summ_json = args.project_root / "results" / "fairness_tables" / det / "fairness_summary.json"
        if not score_csv.exists() or not summ_json.exists():
            rows.append(
                {
                    "detector": det,
                    "status": "missing_artifact",
                }
            )
            continue

        score_df = pd.read_csv(score_csv)
        n_total = int(len(score_df))
        n_test = int((score_df["split"] == "test").sum()) if "split" in score_df.columns else n_total
        groups_test = (
            int(score_df.loc[score_df["split"] == "test", "group"].nunique())
            if "split" in score_df.columns and "group" in score_df.columns
            else int(score_df["group"].nunique())
        )
        is_aligned = n_total == metadata_rows
        status = "current_aligned" if is_aligned else "stale_or_mixed"

        payload = pd.read_json(summ_json, typ="series").to_dict()
        overall = payload.get("overall", {}) if isinstance(payload, dict) else {}
        if not isinstance(overall, dict):
            overall = {}

        rows.append(
            {
                "detector": det,
                "status": status,
                "metadata_rows": metadata_rows,
                "score_rows_total": n_total,
                "score_rows_test": n_test,
                "metadata_groups": metadata_groups,
                "groups_in_test": groups_test,
                "accuracy": safe_float(overall, "accuracy"),
                "auc": safe_float(overall, "auc"),
                "accuracy_disparity": safe_float(overall, "accuracy_disparity"),
                "max_gap_fpr": safe_float(overall, "max_gap_fpr"),
                "max_gap_fnr": safe_float(overall, "max_gap_fnr"),
                "worst_group_error": safe_float(overall, "worst_group_error"),
                "fm_eo_pct": safe_float(overall, "fm_eo_pct"),
                "fdp_pct": safe_float(overall, "fdp_pct"),
                "ffpr_pct": safe_float(overall, "ffpr_pct"),
                "foae_pct": safe_float(overall, "foae_pct"),
                "score_csv_mtime": score_csv.stat().st_mtime,
                "summary_json_mtime": summ_json.stat().st_mtime,
            }
        )

    out_df = pd.DataFrame(rows)
    if len(out_df) > 0 and "summary_json_mtime" in out_df.columns:
        out_df = out_df.sort_values(["status", "summary_json_mtime"], ascending=[True, False]).reset_index(drop=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    notes = [
        "# Latest Run Notes",
        "",
        f"- Metadata rows: **{metadata_rows}**",
        f"- Metadata groups: **{metadata_groups}**",
        "- Status rule: `current_aligned` means detector score rows == current metadata rows.",
        "- If status is `stale_or_mixed`, do not use it as current run evidence in thesis tables.",
        "",
        "## Sample-size caution",
        "- If each group has very few samples (e.g., 1-3 per label), fairness metrics are unstable.",
        "- Prefer at least dozens of samples per group before drawing substantive fairness conclusions.",
    ]
    args.output_notes.write_text("\n".join(notes), encoding="utf-8")

    print(f"[saved] latest overview: {args.output_csv}")
    print(f"[saved] notes: {args.output_notes}")


if __name__ == "__main__":
    main()
