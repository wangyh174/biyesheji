from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor


VALID_GROUPS = [
    "male-doctor",
    "female-doctor",
    "male-nurse",
    "female-nurse",
]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build structured metadata for manually collected real images with AI-suspect audit."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=root / "data" / "real_samples",
        help="Root directory containing the four group folders.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=root / "data" / "real_metadata_auto.csv",
        help="Where to save the auto-labeled metadata CSV.",
    )
    parser.add_argument(
        "--clip-model-id",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="Hugging Face CLIP model ID.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for CLIP inference.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )
    parser.add_argument(
        "--min-width",
        type=int,
        default=512,
        help="Minimum acceptable image width before review_flag is raised.",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=512,
        help="Minimum acceptable image height before review_flag is raised.",
    )
    parser.add_argument(
        "--min-num-pixels",
        type=int,
        default=300000,
        help="Minimum acceptable total pixels before review_flag is raised.",
    )
    parser.add_argument(
        "--min-blur-score",
        type=float,
        default=40.0,
        help="Minimum blur score before review_flag is raised.",
    )
    parser.add_argument(
        "--min-real-photo-conf",
        type=float,
        default=0.40,
        help="Minimum CLIP confidence for real-photo label before AI review is raised.",
    )
    parser.add_argument(
        "--min-ai-margin",
        type=float,
        default=0.05,
        help="If ai_suspect_conf - real_photo_conf exceeds this margin, raise AI review.",
    )
    return parser.parse_args()


class ClipZeroShotTagger:
    def __init__(self, model_id: str, device: str) -> None:
        self.device = device
        self.model = CLIPModel.from_pretrained(model_id).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model.eval()

    def score_labels(self, images: List[Image.Image], labels: List[str]) -> np.ndarray:
        inputs = self.processor(
            text=labels,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        return probs.detach().cpu().numpy().astype(float)


def compute_duplicate_hash(path: Path) -> str:
    img = Image.open(path).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
    arr = np.asarray(img, dtype=np.float32)
    mean_val = float(arr.mean())
    bits = ["1" if x > mean_val else "0" for x in arr.flatten().tolist()]
    return "".join(bits)


def compute_blur_score(path: Path) -> float:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    gy, gx = np.gradient(arr)
    return float(np.var(gx) + np.var(gy))


def collect_image_paths(root: Path) -> List[Tuple[str, Path]]:
    items: List[Tuple[str, Path]] = []
    for group in VALID_GROUPS:
        group_dir = root / group
        if not group_dir.exists():
            continue
        for path in sorted(group_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                items.append((group, path))
    return items


def make_id(group: str, path: Path) -> str:
    digest = hashlib.md5(str(path).encode("utf-8")).hexdigest()[:10]
    prefix = "".join([p[0] for p in group.split("-")])
    return f"{prefix}_{digest}"


def classify_by_clip(
    tagger: ClipZeroShotTagger,
    image_paths: List[Path],
    batch_size: int,
) -> Dict[str, List[str]]:
    face_labels = [
        "a close-up portrait where the face fills most of the image",
        "a chest-up portrait of a single person",
        "a half-body portrait of a single person",
        "a full-body photo of a single person",
    ]
    face_values = ["close", "chest", "half", "full"]

    scene_labels = [
        "a single medical worker in a hospital corridor",
        "a single medical worker in a clinic room",
        "a single medical worker in a hospital ward",
        "a single medical worker at a nursing station",
        "a single medical worker on a plain studio background",
        "a single person in an office-like indoor setting",
        "a single person in some other scene",
    ]
    scene_values = [
        "hospital_corridor",
        "clinic_room",
        "ward",
        "nursing_station",
        "plain_bg",
        "office_like",
        "other",
    ]

    clothing_labels = [
        "a doctor wearing a white coat",
        "a medical worker wearing scrubs",
        "a medical worker in mixed professional medical attire",
        "a person whose clothing is unclear or not obviously medical",
    ]
    clothing_values = ["white_coat", "scrubs", "mixed", "unclear"]

    item_labels = [
        "a visible stethoscope on the person",
        "a visible face mask on the person",
        "a visible hospital badge or ID card on the person",
        "no clearly visible medical accessory",
    ]
    item_values = ["stethoscope", "mask", "badge", "none"]

    real_ai_labels = [
        "a real photograph of a single medical worker",
        "a natural real hospital photo of a person",
        "an ai-generated portrait of a medical worker",
        "a synthetic image of a person",
        "a digitally generated human portrait",
        "a rendered fake person image",
    ]

    results = {
        "face_scale_auto": [],
        "scene_type_auto": [],
        "clothing_type_auto": [],
        "medical_item_auto": [],
        "face_scale_conf": [],
        "scene_type_conf": [],
        "clothing_type_conf": [],
        "medical_item_conf": [],
        "real_photo_conf": [],
        "ai_suspect_conf": [],
        "ai_suspect_score": [],
        "ai_type_auto": [],
    }

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch_images: List[Image.Image] = []
        for p in batch_paths:
            with Image.open(p) as img:
                batch_images.append(img.convert("RGB"))

        face_probs = tagger.score_labels(batch_images, face_labels)
        scene_probs = tagger.score_labels(batch_images, scene_labels)
        clothing_probs = tagger.score_labels(batch_images, clothing_labels)
        item_probs = tagger.score_labels(batch_images, item_labels)
        real_ai_probs = tagger.score_labels(batch_images, real_ai_labels)

        for idx in range(len(batch_paths)):
            face_i = int(np.argmax(face_probs[idx]))
            scene_i = int(np.argmax(scene_probs[idx]))
            clothing_i = int(np.argmax(clothing_probs[idx]))
            item_i = int(np.argmax(item_probs[idx]))

            results["face_scale_auto"].append(face_values[face_i])
            results["scene_type_auto"].append(scene_values[scene_i])
            results["clothing_type_auto"].append(clothing_values[clothing_i])
            results["medical_item_auto"].append(item_values[item_i])

            results["face_scale_conf"].append(float(face_probs[idx][face_i]))
            results["scene_type_conf"].append(float(scene_probs[idx][scene_i]))
            results["clothing_type_conf"].append(float(clothing_probs[idx][clothing_i]))
            results["medical_item_conf"].append(float(item_probs[idx][item_i]))

            real_conf = float(max(real_ai_probs[idx][0], real_ai_probs[idx][1]))
            ai_conf = float(max(real_ai_probs[idx][2], real_ai_probs[idx][3], real_ai_probs[idx][4], real_ai_probs[idx][5]))
            ai_type_idx = int(np.argmax(real_ai_probs[idx][2:]))
            ai_type_values = ["ai_generated_medical_worker", "synthetic_person", "digital_human_portrait", "rendered_fake_person"]
            results["real_photo_conf"].append(real_conf)
            results["ai_suspect_conf"].append(ai_conf)
            results["ai_suspect_score"].append(ai_conf - real_conf)
            results["ai_type_auto"].append(ai_type_values[ai_type_idx])

    return results


def build_review_signals(
    row: pd.Series,
    dup_counts: Dict[str, int],
    min_width: int,
    min_height: int,
    min_num_pixels: int,
    min_blur_score: float,
    min_real_photo_conf: float,
    min_ai_margin: float,
) -> Tuple[int, int, str]:
    reasons = []
    ai_reasons = []

    if int(row["width"]) < min_width:
        reasons.append("low_width")
    if int(row["height"]) < min_height:
        reasons.append("low_height")
    if int(row["num_pixels"]) < min_num_pixels:
        reasons.append("low_pixels")
    if float(row["blur_score"]) < min_blur_score:
        reasons.append("blurry")
    if dup_counts.get(str(row["duplicate_hash"]), 0) > 1:
        reasons.append("duplicate")
    if str(row["face_scale_auto"]) == "full":
        reasons.append("full_body")
    if str(row["scene_type_auto"]) in {"other", "office_like"}:
        reasons.append("scene_review")
    if str(row["clothing_type_auto"]) == "unclear":
        reasons.append("clothing_review")
    if float(row["face_scale_conf"]) < 0.45:
        reasons.append("low_face_conf")
    if float(row["scene_type_conf"]) < 0.45:
        reasons.append("low_scene_conf")
    if float(row["clothing_type_conf"]) < 0.45:
        reasons.append("low_clothing_conf")

    if float(row["real_photo_conf"]) < min_real_photo_conf:
        ai_reasons.append("low_real_photo_conf")
    if float(row["ai_suspect_score"]) > min_ai_margin:
        ai_reasons.append("ai_margin_high")
    if float(row["ai_suspect_conf"]) > float(row["real_photo_conf"]):
        ai_reasons.append("ai_conf_gt_real")

    all_reasons = reasons + ai_reasons
    return (1 if all_reasons else 0, 1 if ai_reasons else 0, ";".join(all_reasons))


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    items = collect_image_paths(args.input_root)
    if not items:
        raise ValueError(f"No images found under: {args.input_root}")

    rows: List[Dict[str, object]] = []

    for group, path in items:
        try:
            with Image.open(path) as img:
                width, height = img.size
            num_pixels = int(width * height)
            blur_score = compute_blur_score(path)
            dup_hash = compute_duplicate_hash(path)

            rows.append(
                {
                    "id": make_id(group, path),
                    "file_path": str(path),
                    "file_name": path.name,
                    "group": group,
                    "width": int(width),
                    "height": int(height),
                    "num_pixels": num_pixels,
                    "blur_score": float(blur_score),
                    "duplicate_hash": dup_hash,
                }
            )
        except (UnidentifiedImageError, OSError, ValueError):
            rows.append(
                {
                    "id": make_id(group, path),
                    "file_path": str(path),
                    "file_name": path.name,
                    "group": group,
                    "width": -1,
                    "height": -1,
                    "num_pixels": -1,
                    "blur_score": -1.0,
                    "duplicate_hash": "read_error",
                    "face_scale_auto": "error",
                    "scene_type_auto": "error",
                    "clothing_type_auto": "error",
                    "medical_item_auto": "error",
                    "face_scale_conf": 0.0,
                    "scene_type_conf": 0.0,
                    "clothing_type_conf": 0.0,
                    "medical_item_conf": 0.0,
                    "real_photo_conf": 0.0,
                    "ai_suspect_conf": 1.0,
                    "ai_suspect_score": 1.0,
                    "ai_type_auto": "read_error",
                    "review_flag": 1,
                    "ai_review_flag": 1,
                    "review_reasons": "read_error",
                    "manual_keep": "",
                    "manual_note": "read_error",
                }
            )

    base_df = pd.DataFrame(rows)
    ok_mask = ~base_df["duplicate_hash"].eq("read_error")
    ok_df = base_df[ok_mask].copy().reset_index(drop=True)

    if len(ok_df) > 0:
        tagger = ClipZeroShotTagger(args.clip_model_id, args.device)
        clip_results = classify_by_clip(
            tagger=tagger,
            image_paths=[Path(p) for p in ok_df["file_path"].tolist()],
            batch_size=args.batch_size,
        )
        for key, values in clip_results.items():
            ok_df[key] = values

        dup_counts = ok_df["duplicate_hash"].value_counts().to_dict()
        signals = ok_df.apply(
            lambda row: build_review_signals(
                row=row,
                dup_counts=dup_counts,
                min_width=args.min_width,
                min_height=args.min_height,
                min_num_pixels=args.min_num_pixels,
                min_blur_score=args.min_blur_score,
                min_real_photo_conf=args.min_real_photo_conf,
                min_ai_margin=args.min_ai_margin,
            ),
            axis=1,
        )
        ok_df[["review_flag", "ai_review_flag", "review_reasons"]] = pd.DataFrame(signals.tolist(), index=ok_df.index)
        ok_df["manual_keep"] = ""
        ok_df["manual_note"] = ""

    final_df = pd.concat([ok_df, base_df[~ok_mask]], ignore_index=True, sort=False)

    desired_cols = [
        "id",
        "file_path",
        "file_name",
        "group",
        "width",
        "height",
        "num_pixels",
        "blur_score",
        "duplicate_hash",
        "face_scale_auto",
        "face_scale_conf",
        "scene_type_auto",
        "scene_type_conf",
        "clothing_type_auto",
        "clothing_type_conf",
        "medical_item_auto",
        "medical_item_conf",
        "real_photo_conf",
        "ai_suspect_conf",
        "ai_suspect_score",
        "ai_type_auto",
        "review_flag",
        "ai_review_flag",
        "review_reasons",
        "manual_keep",
        "manual_note",
    ]

    for col in desired_cols:
        if col not in final_df.columns:
            final_df[col] = ""

    final_df = final_df[desired_cols].sort_values(["group", "file_name"]).reset_index(drop=True)
    final_df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print(f"[saved] metadata: {args.output_csv}")
    print(f"[info] total images: {len(final_df)}")
    print(f"[info] review_flag=1 count: {int((final_df['review_flag'] == 1).sum())}")
    print(f"[info] ai_review_flag=1 count: {int((final_df['ai_review_flag'] == 1).sum())}")


if __name__ == "__main__":
    main()
