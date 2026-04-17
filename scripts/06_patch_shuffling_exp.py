"""
Stage 06: Multi-Scale Patch Shuffling — Structural Attribution Analysis
Reference: Lin et al. (CVPR 2024)

Core idea: Progressively destroy global structure while preserving local texture.
- At 1×1 (original): detector sees full structure → baseline accuracy.
- At 2×2: mild shuffle → slight accuracy drop if detector uses global layout.
- At 4×4, 8×8, 16×16, 32×32: increasing destruction.
- If accuracy barely drops → detector relies on LOCAL texture (frequency artifacts).
- If accuracy drops sharply → detector relies on GLOBAL semantics (risky for fairness).

Output: A CSV + matplotlib curve showing AUC vs. patch_n for each detector.
"""

import os
import sys
import argparse
import subprocess
import numpy as np
import cv2
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from glob import glob
from pathlib import Path


def patch_shuffle(img, patch_size_n=8):
    """
    Splits image into n × n patches and shuffles them randomly.
    512×512 image, n=8 → 64×64 patches.
    """
    h, w, c = img.shape
    patch_h, patch_w = h // patch_size_n, w // patch_size_n

    patches = []
    for i in range(patch_size_n):
        for j in range(patch_size_n):
            patch = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w, :]
            patches.append(patch)

    np.random.shuffle(patches)

    shuffled_img = np.zeros_like(img)
    idx = 0
    for i in range(patch_size_n):
        for j in range(patch_size_n):
            shuffled_img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w, :] = patches[idx]
            idx += 1

    return shuffled_img


def generate_shuffled_images(data_dir, patch_n, seed):
    """Generate shuffled images for a given patch_n and return the output directory."""
    np.random.seed(seed)

    output_dir = os.path.join(data_dir, f"shuffled_{patch_n}x{patch_n}")
    os.makedirs(output_dir, exist_ok=True)

    sources = [
        {"path": os.path.join(data_dir, "generated_raw"), "label": "fake"},
        {"path": os.path.join(data_dir, "real_samples"), "label": "real"}
    ]

    for source in sources:
        if not os.path.exists(source["path"]):
            continue
        groups = [d for d in os.listdir(source["path"])
                  if os.path.isdir(os.path.join(source["path"], d))]
        if source["label"] == "real":
            groups = [d for d in groups if d.endswith("_after")]
        for group in groups:
            group_in = os.path.join(source["path"], group)
            group_out = os.path.join(output_dir, source["label"], group)
            os.makedirs(group_out, exist_ok=True)

            images = glob(os.path.join(group_in, "*.png")) + glob(os.path.join(group_in, "*.jpg"))
            for img_path in images:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                if img.shape[0] != 512 or img.shape[1] != 512:
                    img = cv2.resize(img, (512, 512))
                h, w = img.shape[:2]
                top = (h - 256) // 2
                left = (w - 256) // 2
                img = img[top:top+256, left:left+256]

                shuffled = patch_shuffle(img, patch_n)
                out_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(group_out, out_name), shuffled)

    return output_dir


def run_detector_on_dir(project_root, detector, input_dir, output_dir):
    """Run a detector on a directory of images using 03_run_detectors.py."""
    py = sys.executable
    cmd = [
        py, "scripts/03_run_detectors.py",
        "--project-root", project_root,
        "--detector", detector,
        "--input-dir", input_dir,
        "--output-dir", output_dir,
    ]
    subprocess.run(cmd, cwd=project_root, check=True)


def compute_auc_from_csv(csv_path):
    """Read detector output CSV and compute AUC."""
    if not os.path.exists(csv_path):
        return float("nan")
    df = pd.read_csv(csv_path)
    y = df["y_true"].astype(int).to_numpy()
    score = df["score"].astype(float).to_numpy()
    if len(np.unique(y)) < 2:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, score)


def compute_fpr_gap_from_csv(csv_path):
    """Read detector output CSV and compute max FPR gap across groups."""
    if not os.path.exists(csv_path):
        return float("nan")
    df = pd.read_csv(csv_path)
    fpr_vals = []
    for g, sub in df.groupby("group"):
        y = sub["y_true"].astype(int).to_numpy()
        y_hat = sub["y_hat"].astype(int).to_numpy()
        fp = float(np.sum((y == 0) & (y_hat == 1)))
        tn = float(np.sum((y == 0) & (y_hat == 0)))
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        fpr_vals.append(fpr)
    arr = np.array([v for v in fpr_vals if not np.isnan(v)])
    if len(arr) < 2:
        return float("nan")
    return float(np.max(arr) - np.min(arr))


def plot_structural_attribution_curve(results_df, output_path):
    """
    Plot the Lin (CVPR'24)-style structural attribution curve:
    X-axis: patch_n (structural destruction level)
    Y-axis: AUC (detection accuracy)
    One line per detector.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    detectors = results_df["detector"].unique()
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    markers = ["o", "s", "D", "^"]

    for i, det in enumerate(detectors):
        sub = results_df[results_df["detector"] == det].sort_values("patch_n")
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        # AUC curve
        ax1.plot(sub["patch_n"], sub["auc"], marker=m, color=c,
                 label=det, linewidth=2, markersize=8)

        # FPR Gap curve
        ax2.plot(sub["patch_n"], sub["fpr_gap"], marker=m, color=c,
                 label=det, linewidth=2, markersize=8)

    ax1.set_xlabel("Patch Grid Size (n×n)", fontsize=12)
    ax1.set_ylabel("Detection AUC", fontsize=12)
    ax1.set_title("Structural Attribution: AUC vs. Patch Destruction\n(Lin et al., CVPR 2024)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(results_df["patch_n"].unique())

    ax2.set_xlabel("Patch Grid Size (n×n)", fontsize=12)
    ax2.set_ylabel("Max FPR Gap (Fairness Disparity)", fontsize=12)
    ax2.set_title("Fairness Disparity vs. Structural Destruction\n(BSA, NeurIPS 2024)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(results_df["patch_n"].unique())

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[saved] Structural attribution curve: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Scale Patch Shuffling — Lin CVPR'24 Structural Attribution"
    )
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--patch-scales", type=str, default="1,2,4,8,16,32",
                        help="Comma-separated list of patch grid sizes to test")
    parser.add_argument("--detectors", type=str, default="cnndetection,lgrad,npr",
                        help="Comma-separated list of detectors")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    patch_scales = [int(x.strip()) for x in args.patch_scales.split(",")]
    detectors = [x.strip() for x in args.detectors.split(",")]
    data_dir = os.path.join(args.project_root, "data")
    results_dir = os.path.join(args.project_root, "results", "structural_attribution")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    print("=" * 60)
    print("  Multi-Scale Patch Shuffling Experiment (Lin CVPR'24)")
    print(f"  Scales: {patch_scales}")
    print(f"  Detectors: {detectors}")
    print("=" * 60)

    for patch_n in patch_scales:
        print(f"\n--- Patch Scale: {patch_n}×{patch_n} ---")

        if patch_n == 1:
            # 1×1 means no shuffling = use original images
            shuffled_dir = None
            print("  (Using original images as baseline)")
        else:
            print(f"  Generating shuffled images...")
            shuffled_dir = generate_shuffled_images(data_dir, patch_n, args.seed)

        for det in detectors:
            det_output_dir = os.path.join(results_dir, f"patch_{patch_n}x{patch_n}")
            os.makedirs(det_output_dir, exist_ok=True)

            if patch_n == 1:
                # For baseline, read the original detector scores
                baseline_csv = os.path.join(
                    args.project_root, "results", "detector_outputs", f"{det}_scores.csv"
                )
                auc = compute_auc_from_csv(baseline_csv)
                fpr_gap = compute_fpr_gap_from_csv(baseline_csv)
            else:
                # Run detector on shuffled images
                run_detector_on_dir(args.project_root, det, shuffled_dir, det_output_dir)
                score_csv = os.path.join(det_output_dir, f"{det}_scores.csv")
                auc = compute_auc_from_csv(score_csv)
                fpr_gap = compute_fpr_gap_from_csv(score_csv)

            print(f"  [{det}] AUC={auc:.4f}  FPR_Gap={fpr_gap:.4f}")
            all_results.append({
                "patch_n": patch_n,
                "detector": det,
                "auc": auc,
                "fpr_gap": fpr_gap,
            })

    # Save results table
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(results_dir, "structural_attribution_curve.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n[saved] Results table: {csv_path}")

    # Generate the plot
    plot_path = os.path.join(results_dir, "structural_attribution_curve.png")
    plot_structural_attribution_curve(results_df, plot_path)

    print("\n=== Multi-Scale Patch Shuffling Complete ===")


if __name__ == "__main__":
    main()
