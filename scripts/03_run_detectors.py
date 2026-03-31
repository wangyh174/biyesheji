"""
Step 3: Run representative detector mechanisms (image branch).

Aligned with roadmap:
- CNNDetection: data-driven baseline
- F3Net proxy: frequency-focused features
- LGrad proxy: gradient-texture-focused features
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--metadata-in", type=Path, default=root / "data" / "metadata_balanced.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "detector_outputs")
    parser.add_argument(
        "--detector",
        type=str,
        choices=["cnndetection", "f3net", "lgrad", "gram"],
        default="cnndetection",
    )
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="If set, scan this directory for images instead of using --metadata-in CSV.")
    parser.add_argument("--input-csv", type=Path, default=None,
                        help="Alias for --metadata-in for pipeline compatibility.")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def base_features(image_path: str) -> Tuple[float, float, float, float, float]:
    img = Image.open(image_path).convert("L")
    arr = np.asarray(img, dtype=np.float32) / 255.0

    fft = np.fft.fftshift(np.fft.fft2(arr))
    mag = np.abs(fft)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    yy, xx = np.ogrid[:h, :w]
    mask_low = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    low_energy = mag[mask_low].mean() + 1e-8
    high_energy = mag[~mask_low].mean()
    hf_ratio = float(high_energy / low_energy)

    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=1.2)), dtype=np.float32) / 255.0
    residual = arr - blur
    residual_std = float(np.std(residual))

    gy, gx = np.gradient(arr)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_mean = float(np.mean(grad))
    grad_std = float(np.std(grad))
    pixel_std = float(np.std(arr))
    return hf_ratio, residual_std, grad_mean, grad_std, pixel_std


def select_features(detector: str, feat: Tuple[float, float, float, float, float]) -> List[float]:
    hf_ratio, residual_std, grad_mean, grad_std, pixel_std = feat
    if detector == "cnndetection":
        return [hf_ratio, residual_std, grad_mean, grad_std, pixel_std]
    if detector == "f3net":
        return [hf_ratio, residual_std, pixel_std]
    if detector == "lgrad":
        return [grad_mean, grad_std, residual_std]
    if detector == "gram":
        return [pixel_std, grad_std, hf_ratio, residual_std]
    raise ValueError(f"Unsupported detector: {detector}")


def choose_threshold(y_true: np.ndarray, score: np.ndarray) -> float:
    fpr, tpr, th = roc_curve(y_true, score)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(th[idx])


def build_df_from_dir(input_dir: Path) -> pd.DataFrame:
    """Scan a directory tree for images and build a DataFrame compatible with metadata CSV."""
    rows = []
    for label_dir in sorted(input_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label_name = label_dir.name  # 'fake' or 'real'
        y_true = 1 if label_name == "fake" else 0
        for group_dir in sorted(label_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group = group_dir.name
            for img in sorted(group_dir.glob("*.png")) + sorted(group_dir.glob("*.jpg")):
                rows.append({
                    "id": img.stem,
                    "group": group,
                    "y_true": y_true,
                    "file_path": str(img),
                    "prompt": "",
                })
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    # Handle input source: --input-csv or --input-dir override --metadata-in
    if args.input_csv is not None:
        args.metadata_in = args.input_csv
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_dir is not None:
        df = build_df_from_dir(args.input_dir)
        print(f"[info] Built metadata from directory: {args.input_dir} ({len(df)} images)")
    else:
        df = pd.read_csv(args.metadata_in)
    if len(df) == 0:
        raise ValueError(f"Empty metadata: {args.metadata_in}")

    feat_rows = []
    for p in df["file_path"].tolist():
        feat_rows.append(select_features(args.detector, base_features(str(p))))
    X = np.asarray(feat_rows, dtype=np.float32)
    y = df["y_true"].astype(int).to_numpy()

    stratify_key = df["y_true"].astype(str) + "|" + df["group"].astype(str)
    idx = np.arange(len(df))
    try:
        idx_train, idx_test = train_test_split(
            idx,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=stratify_key,
        )
    except ValueError:
        # For tiny debug runs where some strata have <2 samples, fall back to label-only stratification.
        idx_train, idx_test = train_test_split(
            idx,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )
    split = np.array(["train"] * len(df))
    split[idx_test] = "test"

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[idx_train])
    X_all = scaler.transform(X)

    model = LogisticRegression(max_iter=1500, random_state=args.seed)
    model.fit(X_train, y[idx_train])
    score = model.predict_proba(X_all)[:, 1]
    threshold = choose_threshold(y[idx_train], score[idx_train])
    y_hat = (score >= threshold).astype(int)

    train_auc = roc_auc_score(y[idx_train], score[idx_train])
    test_auc = roc_auc_score(y[idx_test], score[idx_test])

    out = df.copy()
    out["detector_name"] = args.detector
    out["score"] = score
    out["y_hat"] = y_hat
    out["split"] = split

    output_csv = args.output_dir / f"{args.detector}_scores.csv"
    out.to_csv(output_csv, index=False, encoding="utf-8")
    summary_path = args.output_dir / f"{args.detector}_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"detector={args.detector}",
                f"train_auc={train_auc:.6f}",
                f"test_auc={test_auc:.6f}",
                f"threshold={threshold:.6f}",
                f"n_train={len(idx_train)}",
                f"n_test={len(idx_test)}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[saved] detector outputs: {output_csv}")
    print(f"[saved] summary: {summary_path}")
    print(f"[metric] detector={args.detector} train_auc={train_auc:.4f} test_auc={test_auc:.4f}")


if __name__ == "__main__":
    main()
