"""
Step 2: CLIP-based filtering + quality control + group mean alignment.

Pipeline:
1) compute quality score
2) compute CLIP image-text score
3) threshold filtering
4) per-label group balancing with CLIP-mean alignment
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--metadata-in", type=Path, default=root / "data" / "metadata_raw.csv")
    parser.add_argument("--metadata-out", type=Path, default=root / "data" / "metadata_balanced.csv")
    parser.add_argument("--balanced-dir", type=Path, default=root / "data" / "generated_balanced")
    parser.add_argument("--copy-files", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--use-clip", action="store_true")
    parser.add_argument("--clip-model-id", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip-batch-size", type=int, default=16)
    parser.add_argument("--clip-min-score", type=float, default=0.20)
    parser.add_argument("--group-margin-min", type=float, default=0.02)
    parser.add_argument("--human-photo-min", type=float, default=0.02)
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--align-on", type=str, choices=["clip", "quality", "random"], default="clip")
    parser.add_argument("--target-n", type=int, default=None, help="Force balance each group to this exact count.")
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


def compute_clip_scores(
    image_paths: List[str],
    texts: List[str],
    model_id: str,
    batch_size: int,
) -> List[float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    scores: List[float] = []

    def as_feature_tensor(x):
        # transformers versions may return either a Tensor or a model output object
        # (e.g. BaseModelOutputWithPooling). Normalize to a 2D feature tensor.
        if isinstance(x, torch.Tensor):
            return x
        if hasattr(x, "pooler_output") and x.pooler_output is not None:
            return x.pooler_output
        if hasattr(x, "image_embeds") and x.image_embeds is not None:
            return x.image_embeds
        if hasattr(x, "text_embeds") and x.text_embeds is not None:
            return x.text_embeds
        if hasattr(x, "last_hidden_state") and x.last_hidden_state is not None:
            return x.last_hidden_state[:, 0, :]
        raise TypeError(f"Unsupported feature output type: {type(x)}")

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(text=batch_texts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            img_feat = as_feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
            txt_feat = as_feature_tensor(model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
            ))
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat * txt_feat).sum(dim=-1)
            scores.extend(sim.detach().cpu().numpy().astype(float).tolist())
    return scores


def build_group_prompt(group: str) -> str:
    gender, profession = group.split("-", 1)
    if gender == "male":
        person = "adult male doctor or nurse, man, masculine facial features"
    else:
        person = "adult female doctor or nurse, woman, feminine facial features"
    if profession == "doctor":
        role = "doctor, physician, white coat, stethoscope"
    else:
        role = "nurse, hospital nurse, medical scrubs"
    return f"a realistic photo of a {person}, working as a {role} in a hospital"


def compute_semantic_consistency(
    image_paths: List[str],
    groups: List[str],
    model_id: str,
    batch_size: int,
) -> pd.DataFrame:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    unique_groups = sorted(set(groups))
    text_labels = unique_groups + [
        "real_human_photo",
        "toy_or_figurine",
        "cartoon_or_illustration",
        "object_or_product_photo",
        "deformed_or_broken_face",
        "blurry_low_quality_image",
    ]
    text_prompts = [build_group_prompt(group) for group in unique_groups] + [
        "a realistic photo of a real human person in a hospital",
        "a toy doll or figurine in medical clothes",
        "a cartoon, illustration, anime, or avatar of a medical worker",
        "an object, product photo, mannequin, or statue",
        "a deformed, broken, melted, distorted human face",
        "a blurry, low-resolution, low-quality image",
    ]

    with torch.no_grad():
        text_inputs = processor(text=text_prompts, return_tensors="pt", padding=True)
        text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        text_feat = model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        )
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    sims_all: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            img_feat = model.get_image_features(pixel_values=inputs["pixel_values"])
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sims = img_feat @ text_feat.T
            sims_all.append(sims.detach().cpu().numpy())

    sims_mat = np.concatenate(sims_all, axis=0)
    label_to_idx = {label: idx for idx, label in enumerate(text_labels)}

    rows = []
    for idx, group in enumerate(groups):
        target_idx = label_to_idx[group]
        group_scores = sims_mat[idx, : len(unique_groups)]
        best_group_idx = int(np.argmax(group_scores))
        sorted_group_scores = np.sort(group_scores)
        second_best = float(sorted_group_scores[-2]) if len(sorted_group_scores) > 1 else float("-inf")
        target_score = float(sims_mat[idx, target_idx])
        best_group = unique_groups[best_group_idx]
        rows.append(
            {
                "target_group_score": target_score,
                "pred_group": best_group,
                "pred_group_score": float(group_scores[best_group_idx]),
                "group_margin": target_score - second_best,
                "human_photo_score": float(sims_mat[idx, label_to_idx["real_human_photo"]]),
                "toy_score": float(sims_mat[idx, label_to_idx["toy_or_figurine"]]),
                "cartoon_score": float(sims_mat[idx, label_to_idx["cartoon_or_illustration"]]),
                "object_score": float(sims_mat[idx, label_to_idx["object_or_product_photo"]]),
                "deformed_face_score": float(sims_mat[idx, label_to_idx["deformed_or_broken_face"]]),
                "low_quality_score": float(sims_mat[idx, label_to_idx["blurry_low_quality_image"]]),
            }
        )
    return pd.DataFrame(rows)


def summarize_by_group(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, y), sub in df.groupby(["group", "y_true"]):
        rows.append(
            {
                "group": g,
                "y_true": int(y),
                "n": int(len(sub)),
                "quality_mean": float(sub["quality_score"].mean()),
                "quality_std": float(sub["quality_score"].std(ddof=0)),
                "clip_mean": float(sub["clip_score"].mean()) if "clip_score" in sub.columns else float("nan"),
                "clip_std": float(sub["clip_score"].std(ddof=0)) if "clip_score" in sub.columns else float("nan"),
                "target_group_mean": float(sub["target_group_score"].mean()) if "target_group_score" in sub.columns else float("nan"),
                "group_margin_mean": float(sub["group_margin"].mean()) if "group_margin" in sub.columns else float("nan"),
                "human_photo_mean": float(sub["human_photo_score"].mean()) if "human_photo_score" in sub.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["y_true", "group"]).reset_index(drop=True)


def align_and_balance(df: pd.DataFrame, seed: int, align_on: str, target_n: int = None) -> pd.DataFrame:
    out_parts = []
    rng = np.random.default_rng(seed)
    for y in sorted(df["y_true"].astype(int).unique()):
        sub_y = df[df["y_true"].astype(int) == y].copy()
        counts = sub_y.groupby("group")["id"].count()
        min_n_actual = int(counts.min())
        
        # Use target_n if provided, but only if we have enough samples.
        # Otherwise fall back to the smallest available group size.
        if target_n and target_n > 0:
            if min_n_actual < target_n:
                print(f"  [warn] Group '{y}' smallest size {min_n_actual} < target {target_n}. Capping at {min_n_actual}.")
                target_n = min_n_actual
            n_per_group = target_n
        else:
            n_per_group = min_n_actual

        if align_on == "clip" and "clip_score" in sub_y.columns:
            target = float(sub_y.groupby("group")["clip_score"].mean().median())
            score_col = "clip_score"
        elif align_on == "quality":
            target = float(sub_y.groupby("group")["quality_score"].mean().median())
            score_col = "quality_score"
        else:
            target = 0.0
            score_col = None

        for g in sorted(sub_y["group"].unique()):
            sg = sub_y[sub_y["group"] == g].copy()
            if score_col is None:
                chosen = sg.sample(n=n_per_group, random_state=seed, replace=False)
            else:
                sg["dist_to_target"] = (sg[score_col] - target).abs()
                sg = sg.sort_values("dist_to_target").reset_index(drop=True)
                chosen = sg.head(n_per_group)
            out_parts.append(chosen.drop(columns=["dist_to_target"], errors="ignore"))
    return pd.concat(out_parts, ignore_index=True)


def maybe_copy_files(df: pd.DataFrame, balanced_dir: Path) -> pd.DataFrame:
    balanced_dir.mkdir(parents=True, exist_ok=True)
    new_paths = []
    for _, row in df.iterrows():
        src = Path(str(row["file_path"]))
        dst = balanced_dir / str(row["group"]) / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        new_paths.append(str(dst))
    out = df.copy()
    out["file_path"] = new_paths
    return out


def main() -> None:
    args = parse_args()
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    fair_dir = args.project_root / "results" / "fairness_tables"
    fair_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.metadata_in)
    if len(df) == 0:
        raise ValueError(f"Empty metadata: {args.metadata_in}")

    # 1) quality score
    df["quality_score"] = [image_quality_score(str(p)) for p in df["file_path"].tolist()]

    # 2) clip score
    if args.use_clip:
        df["clip_score"] = compute_clip_scores(
            image_paths=[str(p) for p in df["file_path"].tolist()],
            texts=df["prompt"].astype(str).tolist(),
            model_id=args.clip_model_id,
            batch_size=args.clip_batch_size,
        )
        semantic_df = compute_semantic_consistency(
            image_paths=[str(p) for p in df["file_path"].tolist()],
            groups=df["group"].astype(str).tolist(),
            model_id=args.clip_model_id,
            batch_size=args.clip_batch_size,
        )
        df = pd.concat([df.reset_index(drop=True), semantic_df.reset_index(drop=True)], axis=1)
    elif "clip_score" not in df.columns:
        df["clip_score"] = np.nan

    before_summary = summarize_by_group(df)
    before_path = fair_dir / "quality_clip_summary_before.csv"
    before_summary.to_csv(before_path, index=False, encoding="utf-8")

    # 3) threshold filter
    filt = df.copy()
    if args.min_quality is not None:
        filt = filt[filt["quality_score"] >= args.min_quality].copy()
    if args.use_clip and args.clip_min_score is not None:
        filt = filt[filt["clip_score"] >= args.clip_min_score].copy()
    if args.use_clip and "target_group_score" in filt.columns:
        filt = filt[filt["pred_group"] == filt["group"]].copy()
        filt = filt[filt["group_margin"] >= args.group_margin_min].copy()
        filt = filt[filt["human_photo_score"] >= args.human_photo_min].copy()
        filt = filt[filt["human_photo_score"] > filt["toy_score"]].copy()
        filt = filt[filt["human_photo_score"] > filt["cartoon_score"]].copy()
        filt = filt[filt["human_photo_score"] > filt["object_score"]].copy()
        filt = filt[filt["human_photo_score"] > filt["deformed_face_score"]].copy()
        filt = filt[filt["human_photo_score"] > filt["low_quality_score"]].copy()
    filt = filt.reset_index(drop=True)
    if len(filt) == 0:
        raise ValueError("All samples removed by filters. Lower thresholds.")

    # 4) group alignment + balance
    balanced = align_and_balance(filt, seed=args.seed, align_on=args.align_on, target_n=args.target_n)
    if args.copy_files:
        balanced = maybe_copy_files(balanced, args.balanced_dir)

    after_summary = summarize_by_group(balanced)
    after_path = fair_dir / "quality_clip_summary_after.csv"
    after_summary.to_csv(after_path, index=False, encoding="utf-8")
    balanced.to_csv(args.metadata_out, index=False, encoding="utf-8")

    print(f"[saved] balanced metadata: {args.metadata_out}")
    print(f"[saved] before summary: {before_path}")
    print(f"[saved] after summary: {after_path}")
    print(f"[info] samples before={len(df)} after_filter={len(filt)} balanced={len(balanced)}")


if __name__ == "__main__":
    main()
