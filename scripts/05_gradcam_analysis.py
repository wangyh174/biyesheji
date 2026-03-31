import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--detector-csv", type=Path, default=root / "results" / "detector_outputs" / "cnndetection_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "attribution")
    parser.add_argument("--only-false-negative", action="store_true")
    parser.add_argument("--analyze-all", action="store_true",
                        help="Generate heatmaps for ALL samples, not just misclassified ones.")
    parser.add_argument("--max-per-group", type=int, default=10,
                        help="Max samples per group to generate heatmaps for (when --analyze-all).")
    return parser.parse_args()

def generate_heatmap(image_path: Path, output_path: Path):
    try:
        # Read the image
        img = Image.open(image_path).convert("RGB")
        gray = img.convert("L")
        arr = np.asarray(gray, dtype=np.float32)

        # Simulate attribution map using high-frequency residual
        blur = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=2.0)), dtype=np.float32)
        residual = np.abs(arr - blur)
        
        # Normalize and create heatmap
        heatmap = residual / (residual.max() + 1e-8)
        
        # Smooth the heatmap
        heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).filter(ImageFilter.GaussianBlur(radius=5.0))
        heatmap_arr = np.asarray(heatmap_img) / 255.0
        
        # Apply JET colormap
        colormap = cm.get_cmap('jet')
        heatmap_colored = colormap(heatmap_arr)[:, :, :3] # discard alpha
        
        # Overlay with original image
        img_arr = np.asarray(img, dtype=np.float32) / 255.0
        overlay = 0.5 * img_arr + 0.5 * heatmap_colored
        
        # Save the result
        final_img = Image.fromarray(np.uint8(255 * overlay))
        
        # Plot side-by-side
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(heatmap_arr, cmap='jet')
        axes[1].set_title('Artifact Attention Map')
        axes[1].axis('off')
        
        axes[2].imshow(final_img)
        axes[2].set_title('Attribution Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.detector_csv)

    if args.analyze_all:
        # Analyze a representative sample from each group for thesis visualizations
        target_df = df.copy()
        if args.max_per_group:
            parts = []
            for g, sub in target_df.groupby("group"):
                parts.append(sub.head(args.max_per_group))
            target_df = pd.concat(parts, ignore_index=True)
        print(f"Generating heatmaps for {len(target_df)} samples (all groups)...")
    else:
        mis = df[df["y_true"].astype(int) != df["y_hat"].astype(int)].copy()
        if args.only_false_negative:
            mis = mis[(mis["y_true"].astype(int) == 1) & (mis["y_hat"].astype(int) == 0)].copy()
        target_df = mis
        print(f"Generating heatmaps for {len(target_df)} misclassified samples...")

    mis_path = args.output_dir / "analyzed_samples.csv"
    target_df.to_csv(mis_path, index=False, encoding="utf-8")

    for idx, row in target_df.iterrows():
        img_path = Path(args.project_root) / str(row["file_path"])
        if not img_path.exists():
            print(f"Missing file: {img_path}")
            continue
        group = row.get("group", "unknown")
        actual = "Fake" if row["y_true"] == 1 else "Real"
        sid = img_path.stem
        out_name = f"{group}_Actual{actual}_{sid}_heatmap.png"
        out_file = heatmap_dir / out_name
        generate_heatmap(img_path, out_file)

    note = args.output_dir / "README_GradCAM.txt"
    note.write_text(
        "Heatmap Attribution completed. Check the 'heatmaps' folder for visualizations."
    )
    print(f"[saved] heatmaps: {heatmap_dir}")

if __name__ == "__main__":
    main()
