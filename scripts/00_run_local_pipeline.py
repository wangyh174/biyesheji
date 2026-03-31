"""
Master Thesis Pipeline Orchestrator (Stages 01-09)
Aligned with: Fair-Diffusion, BSA (NeurIPS 24), D3 (ICCV 25), Lin (CVPR 24)
"""
import argparse
import subprocess
import sys
import os
from pathlib import Path

def run_cmd(script_name, args, cwd):
    py = sys.executable
    cmd = [py, f"scripts/{script_name}"] + args
    print(f"\n>>> Running {script_name} {' '.join(args)}...")
    subprocess.run(cmd, cwd=cwd, check=True)

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=str, default=str(root))
    parser.add_argument("--real-source", type=str, default="local", choices=["local", "diffusers", "mock"])
    parser.add_argument("--detectors", type=str, default="cnndetection,f3net")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE")
    args = parser.parse_args()

    project_root = args.project_root
    detector_list = [d.strip() for d in args.detectors.split(",")]

    # --- 01: Data Generation ---
    run_cmd("01_generate.py", [
        "--project-root", project_root,
        "--real-source", args.real_source,
        "--samples-per-group", str(args.samples),
        "--model-id", args.model_id,
        "--generator", "fairdiffusion",
        "--seed", str(args.seed)
    ], project_root)

    # --- 01b: Generation Audit ---
    run_cmd("01b_generation_audit.py", ["--project-root", project_root], project_root)

    # --- 02: Quality Filtering ---
    run_cmd("02_quality_filter.py", [
        "--project-root", project_root,
        "--use-clip", "True",
        "--clip-min-score", "0.22"
    ], project_root)

    # --- 03 & 04: Baseline Evaluation ---
    for det in detector_list:
        run_cmd("03_run_detectors.py", [
            "--project-root", project_root,
            "--detector", det,
            "--input-dir", "data/generated_raw"
        ], project_root)
    
    run_cmd("04_fairness_eval.py", ["--project-root", project_root], project_root)

    for det in detector_list:
        run_cmd("05_gradcam_analysis.py", ["--project-root", project_root, "--detector", det], project_root)

    # --- 06: Structural Attribution ---
    run_cmd("06_patch_shuffling_exp.py", ["--project-root", project_root, "--patch-n", "8"], project_root)
    for det in detector_list:
        run_cmd("03_run_detectors.py", [
            "--project-root", project_root,
            "--detector", det,
            "--input-dir", "data/shuffled_8x8"
        ], project_root)

    # --- 07: Physical Consistency ---
    run_cmd("07_physical_consistency.py", ["--project-root", project_root, "--max-samples", "100"], project_root)

    # --- 08: Innovation ---
    run_cmd("08_high_pass_innovation.py", ["--project-root", project_root, "--process-all"], project_root)
    for det in detector_list:
        run_cmd("03_run_detectors.py", [
            "--project-root", project_root,
            "--detector", det,
            "--input-dir", "data/high_pass_residuals"
        ], project_root)

    # --- 09: Master Report ---
    run_cmd("09_master_report.py", ["--project-root", project_root], project_root)

if __name__ == "__main__":
    main()
