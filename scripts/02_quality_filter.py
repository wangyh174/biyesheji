"""
Step 2: CLIP-based filtering + quality control + group mean alignment.

Pipeline:
1) compute quality score
2) compute CLIP image-text score
3) threshold filtering
4) per-label group balancing with CLIP-mean alignment

Thesis note:
- generation already applies fairness-oriented semantic control
- this stage should remove obviously bad samples without reintroducing
  strong gendered filtering pressure
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
from tqdm.auto import tqdm
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
    parser.add_argument(
        "--clip-text-mode",
        type=str,
        choices=["metadata", "profession"],
        default="profession",
        help=(
            "Text source for CLIP image-text scoring. "
            "'profession' uses a more neutral profession-centered prompt and is safer "
            "for fairness-sensitive filtering."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Inference device for CLIP-based filtering. Default is cuda.",
    )
    parser.add_argument("--clip-min-score", type=float, default=0.20)
    parser.add_argument("--group-margin-min", type=float, default=0.02)
    parser.add_argument("--human-photo-min", type=float, default=0.02)
    parser.add_argument(
        "--disable-toy-check",
        action="store_true",
        help="Do not require human_photo_score > toy_score.",
    )
    parser.add_argument(
        "--disable-cartoon-check",
        action="store_true",
        help="Do not require human_photo_score > cartoon_score.",
    )
    parser.add_argument(
        "--strict-group-check",
        action="store_true",
        help=(
            "Require semantic classifier predicted group to exactly match metadata group. "
            "Disabled by default to avoid over-filtering fairness-controlled generations."
        ),
    )
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--align-on", type=str, choices=["clip", "quality", "random"], default="clip")
    parser.add_argument("--target-n", type=int, default=None, help="Force balance each group to this exact count.")
    parser.add_argument(
        "--reuse-scored",
        action="store_true",
        default=True,
        help="Reuse data/metadata_scored.csv when it matches current input rows and file paths.",
    )
    parser.add_argument(
        "--no-reuse-scored",
        dest="reuse_scored",
        action="store_false",
        help="Force recomputing quality and CLIP scores from scratch.",
    )
    return parser.parse_args()


def validate_runtime(device: str, use_clip: bool) -> None:
    if use_clip and device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested for CLIP filtering, but no GPU is available. "
            "In Colab, switch Runtime -> Change runtime type -> GPU, "
            "or pass --device cpu if you intentionally want CPU mode."
        )


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


def compute_clip_scores(
    image_paths: List[str],
    texts: List[str],
    model_id: str,
    batch_size: int,
    device: str,
) -> List[float]:
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)
    model.eval()

    scores: List[float] = []

    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, len(image_paths), batch_size),
            total=total_batches,
            desc="CLIP text-image scoring",
            leave=False,
        ):
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
        person = "adult person presenting as male"
    else:
        person = "adult person presenting as female"
    if profession == "doctor":
        role = "doctor, physician, white coat, stethoscope"
    else:
        role = "nurse, hospital nurse, medical scrubs"
    return f"a realistic single-person hospital portrait of a {person}, working as a {role}"


def build_profession_prompt(group: str) -> str:
    _, profession = group.split("-", 1)
    if profession == "doctor":
        role = "doctor, physician, white coat, stethoscope"
    else:
        role = "nurse, hospital nurse, medical scrubs"
    return f"a realistic single-person portrait photo of a {role} in a hospital"


def build_clip_score_texts(df: pd.DataFrame, clip_text_mode: str) -> List[str]:
    if clip_text_mode == "metadata":
        return df["prompt"].astype(str).tolist()
    return [build_profession_prompt(group) for group in df["group"].astype(str).tolist()]


def compute_semantic_consistency(
    image_paths: List[str],
    groups: List[str],
    model_id: str,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
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
        text_feat = as_feature_tensor(model.get_text_features(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
        ))
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    sims_all: List[np.ndarray] = []
    total_batches = (len(image_paths) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in tqdm(
            range(0, len(image_paths), batch_size),
            total=total_batches,
            desc="CLIP semantic consistency",
            leave=False,
        ):
            batch_paths = image_paths[i : i + batch_size]
            images = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = processor(images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            img_feat = as_feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
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


def build_filter_audit(df_before: pd.DataFrame, df_filtered: pd.DataFrame, df_balanced: pd.DataFrame) -> pd.DataFrame:
    def count_df(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        if len(df) == 0:
            return pd.DataFrame(columns=["group", "y_true", col_name])
        out = (
            df.groupby(["group", "y_true"], dropna=False)["id"]
            .count()
            .reset_index()
            .rename(columns={"id": col_name})
        )
        out["y_true"] = out["y_true"].astype(int)
        return out

    before = count_df(df_before, "n_before")
    filtered = count_df(df_filtered, "n_after_filter")
    balanced = count_df(df_balanced, "n_balanced")

    audit = before.merge(filtered, on=["group", "y_true"], how="outer")
    audit = audit.merge(balanced, on=["group", "y_true"], how="outer")
    audit[["n_before", "n_after_filter", "n_balanced"]] = (
        audit[["n_before", "n_after_filter", "n_balanced"]].fillna(0).astype(int)
    )

    before_nonzero = audit["n_before"].replace(0, np.nan)
    after_filter_nonzero = audit["n_after_filter"].replace(0, np.nan)
    audit["filter_keep_rate"] = (audit["n_after_filter"] / before_nonzero).fillna(0.0)
    audit["filter_drop_rate"] = 1.0 - audit["filter_keep_rate"]
    audit["balance_keep_rate_from_filtered"] = (audit["n_balanced"] / after_filter_nonzero).fillna(0.0)
    audit["balance_keep_rate_from_raw"] = (audit["n_balanced"] / before_nonzero).fillna(0.0)
    return audit.sort_values(["y_true", "group"]).reset_index(drop=True)


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
    validate_runtime(args.device, args.use_clip)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)
    fair_dir = args.project_root / "results" / "fairness_tables"
    fair_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.project_root / "data" / "metadata_scored.csv"
    filtered_prebalance_path = args.project_root / "data" / "metadata_filtered_prebalance.csv"
    print(f"[info] running on device: {args.device}")
    print(
        f"[info] clip_text_mode={args.clip_text_mode} strict_group_check={args.strict_group_check} "
        f"align_on={args.align_on}"
    )

    df = pd.read_csv(args.metadata_in)
    if len(df) == 0:
        raise ValueError(f"Empty metadata: {args.metadata_in}")

    print(f"[stage] loaded metadata rows={len(df)} from {args.metadata_in}")
    reused = False
    if args.reuse_scored and scored_path.exists():
        print(f"[stage] checking reusable scored cache: {scored_path}")
        cached = pd.read_csv(scored_path)
        same_len = len(cached) == len(df)
        same_paths = same_len and cached["file_path"].astype(str).tolist() == df["file_path"].astype(str).tolist()
        if same_len and same_paths:
            required_cols = {"quality_score"}
            if args.use_clip:
                required_cols.update(
                    {
                        "clip_score",
                        "target_group_score",
                        "pred_group",
                        "pred_group_score",
                        "group_margin",
                        "human_photo_score",
                        "toy_score",
                        "cartoon_score",
                        "object_score",
                        "deformed_face_score",
                        "low_quality_score",
                    }
                )
            missing_cols = [c for c in required_cols if c not in cached.columns]
            if not missing_cols:
                df = cached
                reused = True
                print(f"[done] reused scored cache with {len(df)} rows")
            else:
                print(f"[warn] scored cache missing columns: {missing_cols}; recomputing")
        else:
            print("[warn] scored cache does not match current metadata; recomputing")

    if not reused:
        # 1) quality score
        print("[stage] computing quality_score for all images")
        df["quality_score"] = [
            image_quality_score(str(p))
            for p in tqdm(df["file_path"].tolist(), desc="Quality scoring", leave=False)
        ]
        print("[done] quality_score computed")

        # 2) clip score
        if args.use_clip:
            print(
                f"[stage] computing CLIP scores model={args.clip_model_id} "
                f"batch_size={args.clip_batch_size}"
            )
            clip_texts = build_clip_score_texts(df, args.clip_text_mode)
            df["clip_score"] = compute_clip_scores(
                image_paths=[str(p) for p in df["file_path"].tolist()],
                texts=clip_texts,
                model_id=args.clip_model_id,
                batch_size=args.clip_batch_size,
                device=args.device,
            )
            print("[done] CLIP text-image scores computed")

            print("[stage] computing CLIP semantic consistency scores")
            semantic_df = compute_semantic_consistency(
                image_paths=[str(p) for p in df["file_path"].tolist()],
                groups=df["group"].astype(str).tolist(),
                model_id=args.clip_model_id,
                batch_size=args.clip_batch_size,
                device=args.device,
            )
            df = pd.concat([df.reset_index(drop=True), semantic_df.reset_index(drop=True)], axis=1)
            print("[done] CLIP semantic consistency computed")
        elif "clip_score" not in df.columns:
            df["clip_score"] = np.nan

    before_summary = summarize_by_group(df)
    before_path = fair_dir / "quality_clip_summary_before.csv"
    before_summary.to_csv(before_path, index=False, encoding="utf-8")
    df.to_csv(scored_path, index=False, encoding="utf-8")

    print("[stage] applying threshold filters")
    # 3) threshold filter
    filt = df.copy()
    if args.min_quality is not None:
        filt = filt[filt["quality_score"] >= args.min_quality].copy()
    if args.use_clip and args.clip_min_score is not None:
        filt = filt[filt["clip_score"] >= args.clip_min_score].copy()
    if args.use_clip and "target_group_score" in filt.columns:
        if args.strict_group_check:
            filt = filt[filt["pred_group"] == filt["group"]].copy()
        filt = filt[filt["group_margin"] >= args.group_margin_min].copy()
        filt = filt[filt["human_photo_score"] >= args.human_photo_min].copy()
        if not args.disable_toy_check:
            filt = filt[filt["human_photo_score"] > filt["toy_score"]].copy()
        if not args.disable_cartoon_check:
            filt = filt[filt["human_photo_score"] > filt["cartoon_score"]].copy()
        filt = filt[filt["human_photo_score"] > filt["object_score"]].copy()
        filt = filt[filt["human_photo_score"] > filt["deformed_face_score"]].copy()
        filt = filt[filt["human_photo_score"] > filt["low_quality_score"]].copy()
    filt = filt.reset_index(drop=True)
    if len(filt) == 0:
        raise ValueError("All samples removed by filters. Lower thresholds.")
    filt.to_csv(filtered_prebalance_path, index=False, encoding="utf-8")
    print(f"[done] threshold filtering kept {len(filt)} / {len(df)} rows")

    # 4) group alignment + balance
    print(f"[stage] aligning and balancing by {args.align_on}")
    balanced = align_and_balance(filt, seed=args.seed, align_on=args.align_on, target_n=args.target_n)
    print(f"[done] balancing produced {len(balanced)} rows")
    if args.copy_files:
        print(f"[stage] copying balanced files to {args.balanced_dir}")
        balanced = maybe_copy_files(balanced, args.balanced_dir)
        print("[done] balanced files copied")

    after_summary = summarize_by_group(balanced)
    after_path = fair_dir / "quality_clip_summary_after.csv"
    after_summary.to_csv(after_path, index=False, encoding="utf-8")
    audit_summary = build_filter_audit(df_before=df, df_filtered=filt, df_balanced=balanced)
    audit_path = fair_dir / "quality_clip_filter_audit.csv"
    audit_summary.to_csv(audit_path, index=False, encoding="utf-8")
    balanced.to_csv(args.metadata_out, index=False, encoding="utf-8")

    print(f"[saved] balanced metadata: {args.metadata_out}")
    print(f"[saved] scored metadata: {scored_path}")
    print(f"[saved] filtered prebalance metadata: {filtered_prebalance_path}")
    print(f"[saved] before summary: {before_path}")
    print(f"[saved] after summary: {after_path}")
    print(f"[saved] filter audit: {audit_path}")
    print(f"[info] samples before={len(df)} after_filter={len(filt)} balanced={len(balanced)}")


if __name__ == "__main__":
    main()
