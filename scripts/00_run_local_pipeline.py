"""
One-command local pipeline aligned with thesis roadmap:
1) Prompt/AIGC generation + cleaning
2) Multi-detector evaluation + fairness metrics
3) Grad-CAM attribution entrypoint
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(args: list[str], cwd: Path) -> None:
    print("[run]", " ".join(args))
    subprocess.run(args, cwd=str(cwd), check=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--generator", type=str, choices=["mock", "diffusers", "fairdiffusion"], default="mock")
    parser.add_argument("--samples-per-group", type=int, default=30)
    parser.add_argument("--real-per-group", type=int, default=30)
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--bootstrap-iters", type=int, default=300)
    parser.add_argument("--detectors", type=str, default="cnndetection,f3net,lgrad")
    parser.add_argument("--run-gradcam", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-min-score", type=float, default=0.20)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--real-model-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.project_root
    scripts = root / "scripts"
    py = sys.executable
    common = ["--project-root", str(root)]

    # Stage 1: data preparation (no race, only gender x profession)
    cmd1 = [
        py,
        str(scripts / "01_generate.py"),
        *common,
        "--generator",
        args.generator,
        "--samples-per-group",
        str(args.samples_per_group),
        "--mock-real-per-group",
        str(args.real_per_group),
        "--genders",
        "male,female",
        "--professions",
        "doctor,nurse",
        "--seed",
        str(args.seed),
        "--overwrite",
    ]
    if args.generator in ("diffusers", "fairdiffusion"):
        cmd1 += ["--width", "512", "--height", "512", "--steps", "30"]
    if args.model_path is not None:
        cmd1 += ["--model-path", str(args.model_path)]
    if args.real_model_path is not None:
        cmd1 += ["--real-model-path", str(args.real_model_path)]

    cmd2 = [
        py,
        str(scripts / "02_quality_filter.py"),
        *common,
        "--seed",
        str(args.seed),
        "--use-clip",
        "--clip-min-score",
        str(args.clip_min_score),
        "--align-on",
        "clip",
    ]
    cmd_audit = [
        py,
        str(scripts / "01b_generation_audit.py"),
        *common,
        "--metadata-in",
        str(root / "data" / "metadata_raw.csv"),
        "--save-controlled-metadata",
        "--controlled-metadata-out",
        str(root / "data" / "metadata_gen_control.csv"),
        "--seed",
        str(args.seed),
    ]
    cmd2 += ["--metadata-in", str(root / "data" / "metadata_gen_control.csv")]
    if args.min_quality is not None:
        cmd2 += ["--min-quality", str(args.min_quality)]

    run_cmd(cmd1, cwd=root)
    run_cmd(cmd_audit, cwd=root)
    run_cmd(cmd2, cwd=root)

    # Stage 2: model evaluation + fairness metrics
    detector_list = [x.strip() for x in args.detectors.split(",") if x.strip()]
    for det in detector_list:
        cmd3 = [
            py,
            str(scripts / "03_run_detectors.py"),
            *common,
            "--detector",
            det,
            "--seed",
            str(args.seed),
        ]
        run_cmd(cmd3, cwd=root)

        det_csv = root / "results" / "detector_outputs" / f"{det}_scores.csv"
        det_out_dir = root / "results" / "fairness_tables" / det
        cmd4 = [
            py,
            str(scripts / "04_fairness_eval.py"),
            "--detector-csv",
            str(det_csv),
            "--output-dir",
            str(det_out_dir),
            "--split",
            "test",
            "--bootstrap-iters",
            str(args.bootstrap_iters),
            "--seed",
            str(args.seed),
        ]
        run_cmd(cmd4, cwd=root)

    # Stage 3: attribution entry (optional)
    if args.run_gradcam:
        cmd5 = [py, str(scripts / "05_gradcam_analysis.py"), *common]
        run_cmd(cmd5, cwd=root)

    # Stage 4: consolidate latest-run outputs for thesis/report writing.
    cmd6 = [
        py,
        str(scripts / "06_consolidate_results.py"),
        *common,
        "--detectors",
        ",".join(detector_list),
    ]
    run_cmd(cmd6, cwd=root)

    print("[done] Local roadmap pipeline completed.")
    print(f"[results] {root / 'results' / 'fairness_tables'}")


if __name__ == "__main__":
    main()
