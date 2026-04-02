"""
Master Thesis Pipeline Orchestrator (Stages 01-09)
Aligned with: Fair-Diffusion (CVPR'24), BSA (NeurIPS'24), D3 (ICCV'25), Lin (CVPR'24)
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
    parser.add_argument("--detectors", type=str, default="cnndetection,f3net,gram,lgrad")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-id", type=str, default="SG161222/Realistic_Vision_V5.1_noVAE")
    parser.add_argument("--buffer-extra", type=int, default=20)
    args = parser.parse_args()

    project_root = args.project_root
    detector_list = [d.strip() for d in args.detectors.split(",")]
    buffer = args.samples + args.buffer_extra

    # === Stage 01: Data Generation (Fair-Diffusion) ===
    run_cmd("01_generate.py", [
        "--project-root", project_root,
        "--real-source", args.real_source,
        "--samples-per-group", str(buffer),       # Overgenerate for quality filtering
        "--mock-real-per-group", str(buffer),      # Fix #9: Keep fake/real symmetric
        "--model-id", args.model_id,
        "--generator", "fairdiffusion",
        "--seed", str(args.seed)
    ], project_root)

    # === Stage 01b: Generation Audit ===
    run_cmd("01b_generation_audit.py", ["--project-root", project_root], project_root)

    # === Stage 02: Quality Filtering & Elite Selection (CLIP 0.22) ===
    run_cmd("02_quality_filter.py", [
        "--project-root", project_root,
        "--use-clip",
        "--clip-min-score", "0.22",
        "--group-margin-min", "0.03",
        "--human-photo-min", "0.02",
        "--align-on", "clip",
        "--target-n", str(args.samples)  # Hard-cap at exactly N highest-quality samples
    ], project_root)

    # All downstream stages use the balanced (QC-passed) metadata
    metadata_balanced = str(Path(project_root) / "data" / "metadata_balanced.csv")

    # === Stage 03: Run Detectors (CNNDetection, F3Net, Gram, LGrad) ===
    for det in detector_list:
        run_cmd("03_run_detectors.py", [
            "--project-root", project_root,
            "--detector", det,
            "--metadata-in", metadata_balanced      # Fix #4: Use correct param name
        ], project_root)

    # === Stage 04: Fairness Evaluation (per-detector) ===
    # Fix #6: Run fairness eval for EACH detector separately
    fairness_dir = Path(project_root) / "results" / "fairness_tables"
    for det in detector_list:
        det_csv = str(Path(project_root) / "results" / "detector_outputs" / f"{det}_scores.csv")
        det_fair_dir = str(fairness_dir / det)
        run_cmd("04_fairness_eval.py", [
            "--detector-csv", det_csv,
            "--output-dir", det_fair_dir
        ], project_root)

    # === Stage 05: Visual Attribution (Grad-CAM heatmaps) ===
    for det in detector_list:
        det_csv = str(Path(project_root) / "results" / "detector_outputs" / f"{det}_scores.csv")
        run_cmd("05_gradcam_analysis.py", [
            "--project-root", project_root,
            "--detector-csv", det_csv
        ], project_root)

    # === Stage 06a: Multi-Scale Structural Attribution (Lin CVPR'24) ===
    # Tests patch scales 1×1 (baseline) through 32×32, plots AUC decay curve
    run_cmd("06_patch_shuffling_exp.py", [
        "--project-root", project_root,
        "--patch-scales", "1,2,4,8,16,32",
        "--detectors", args.detectors,
        "--seed", str(args.seed)
    ], project_root)

    # === Stage 06b: Consolidate Results ===
    run_cmd("06_consolidate_results.py", [
        "--project-root", project_root,
        "--detectors", args.detectors
    ], project_root)

    # === Stage 07: Physical Consistency (GLCM — D3 ICCV'25) ===
    run_cmd("07_physical_consistency.py", [
        "--project-root", project_root,
        "--max-samples", str(buffer)
    ], project_root)

    # === Stage 08: Innovation - Semantic-Noise Decoupling (High-pass Residuals) ===
    run_cmd("08_high_pass_innovation.py", [
        "--project-root", project_root,
        "--process-all"
    ], project_root)

    # Run detectors on high-pass residuals to test noise-only detection
    for det in detector_list:
        run_cmd("03_run_detectors.py", [
            "--project-root", project_root,
            "--detector", det,
            "--input-dir", str(Path(project_root) / "data" / "high_pass_residuals"),
            "--output-dir", str(Path(project_root) / "results" / "detector_outputs_highpass")
        ], project_root)

    # === Stage 09: Master Report ===
    run_cmd("09_master_report.py", [
        "--project-root", project_root,
        "--detectors", args.detectors
    ], project_root)

    print("\n" + "=" * 60)
    print("  ALL STAGES COMPLETE — Results in results/ directory")
    print("=" * 60)

if __name__ == "__main__":
    main()
