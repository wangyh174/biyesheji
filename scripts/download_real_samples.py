"""
Download real photographs from free stock image APIs + CLIP-based filtering
for precise gender×profession matching.

Pipeline:
  1. Search Pexels/Pixabay/Unsplash with broad medical queries
  2. Download candidates (more than needed)
  3. Use CLIP to score each image against precise description
  4. Only keep images above CLIP similarity threshold
  5. Save top-scoring images per group

Usage:
  python scripts/download_real_samples.py --per-group 50 --pexels-key "YOUR_KEY"
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from PIL import Image


# ─── CLIP descriptions for precise filtering ────────────────────────────────
# These are the "judge sentences" CLIP will compare each image against
CLIP_DESCRIPTIONS: Dict[str, str] = {
    "male-doctor": "a photo of a male doctor wearing a white coat with a stethoscope in a hospital",
    "female-doctor": "a photo of a female doctor wearing a white coat with a stethoscope in a hospital",
    "male-nurse": "a photo of a male nurse wearing scrubs in a hospital",
    "female-nurse": "a photo of a female nurse wearing scrubs in a hospital",
}

# Negative descriptions to reject mismatches
CLIP_NEGATIVES: Dict[str, List[str]] = {
    "male-doctor": ["a photo of a female doctor", "a photo of a nurse", "a photo of a patient in hospital"],
    "female-doctor": ["a photo of a male doctor", "a photo of a nurse", "a photo of a patient in hospital"],
    "male-nurse": ["a photo of a female nurse", "a photo of a doctor", "a photo of a patient in hospital"],
    "female-nurse": ["a photo of a male nurse", "a photo of a doctor", "a photo of a patient in hospital"],
}

# ─── search queries ─────────────────────────────────────────────────────────
# Broader queries to get more candidates, CLIP will do the precise filtering
SEARCH_QUERIES: Dict[str, List[str]] = {
    "male-doctor": [
        "male doctor hospital portrait",
        "man doctor stethoscope white coat",
        "male physician medical professional",
        "man doctor clinic portrait",
        "male surgeon hospital professional",
        "man healthcare doctor uniform",
        "male doctor examining patient",
        "young male doctor portrait",
        "man doctor office professional",
        "male doctor medical practice",
        "male doctor",
        "man physician",
    ],
    "female-doctor": [
        "female doctor hospital portrait",
        "woman doctor stethoscope white coat",
        "female physician medical professional",
        "woman doctor clinic portrait",
        "female surgeon hospital professional",
        "woman healthcare doctor uniform",
        "female doctor examining patient",
        "young female doctor portrait",
        "woman doctor office professional",
        "female doctor medical practice",
        "female doctor",
        "woman physician",
    ],
    "male-nurse": [
        "male nurse hospital portrait",
        "man nurse scrubs uniform",
        "male nurse caring patient",
        "man nursing professional hospital",
        "male nurse medical scrubs",
        "man registered nurse portrait",
        "male nurse healthcare worker",
        "young male nurse hospital",
        "man nurse clinic uniform",
        "male nurse ward hospital",
        "male nurse",
        "man nurse scrubs",
    ],
    "female-nurse": [
        "female nurse hospital portrait",
        "woman nurse scrubs uniform",
        "female nurse caring patient",
        "woman nursing professional hospital",
        "female nurse medical scrubs",
        "woman registered nurse portrait",
        "female nurse healthcare worker",
        "young female nurse hospital",
        "woman nurse clinic uniform",
        "female nurse ward hospital",
        "female nurse",
        "woman nurse scrubs",
    ],
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download real stock photos with CLIP filtering for fairness evaluation"
    )
    parser.add_argument("--output-dir", type=Path, default=root / "data" / "real_samples")
    parser.add_argument("--per-group", type=int, default=50)
    parser.add_argument("--size", type=int, default=512, help="Resize to NxN")
    parser.add_argument("--clip-threshold", type=float, default=0.22,
                        help="Minimum CLIP similarity to keep an image")
    parser.add_argument("--pexels-key", type=str, default=os.environ.get("PEXELS_API_KEY", ""))
    parser.add_argument("--pixabay-key", type=str, default=os.environ.get("PIXABAY_API_KEY", ""))
    parser.add_argument("--unsplash-key", type=str, default=os.environ.get("UNSPLASH_ACCESS_KEY", ""))
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
#  CLIP filter
# ═══════════════════════════════════════════════════════════════════════════
_clip_model = None
_clip_processor = None


def _load_clip():
    global _clip_model, _clip_processor
    if _clip_model is not None:
        return _clip_model, _clip_processor

    import torch
    from transformers import CLIPModel, CLIPProcessor

    print("[CLIP] Loading openai/clip-vit-base-patch32 ...")
    _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model = _clip_model.to(device).eval()
    print(f"[CLIP] Loaded on {device}")
    return _clip_model, _clip_processor


def clip_check(img: Image.Image, group: str, threshold: float) -> Tuple[bool, float]:
    """
    Check if image matches the group description using CLIP.
    Returns (pass, score).
    The image must score higher on the target description than ALL negative descriptions.
    """
    import torch

    model, processor = _load_clip()
    device = next(model.parameters()).device

    target_text = CLIP_DESCRIPTIONS[group]
    negative_texts = CLIP_NEGATIVES[group]
    all_texts = [target_text] + negative_texts

    inputs = processor(
        text=all_texts,
        images=img,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image[0]  # shape: [num_texts]
        probs = logits.softmax(dim=0)

    target_prob = probs[0].item()
    target_score = logits[0].item() / 100.0  # normalize to ~0-1 range

    # Must be the highest scoring AND above absolute threshold
    is_best = probs[0] == probs.max()
    passes = bool(is_best and target_score >= threshold)

    return passes, target_prob


# ═══════════════════════════════════════════════════════════════════════════
#  Image utilities
# ═══════════════════════════════════════════════════════════════════════════
def _download_image(url: str, timeout: int = 15) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception:
        return None


def _content_hash(img: Image.Image) -> str:
    thumb = img.resize((16, 16)).convert("L")
    return hashlib.md5(thumb.tobytes()).hexdigest()


def _save_image(img: Image.Image, out_dir: Path, group: str, idx: int, size: int) -> Path:
    w, h = img.size
    s = min(w, h)
    left = (w - s) // 2
    top = (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    img = img.resize((size, size), Image.LANCZOS)
    out_path = out_dir / group / f"real_{group}_{idx:04d}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════
#  API sources
# ═══════════════════════════════════════════════════════════════════════════
def fetch_pexels(query: str, api_key: str, per_page: int = 40, page: int = 1) -> List[str]:
    if not api_key:
        return []
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page, "page": page}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return [p["src"]["large"] for p in resp.json().get("photos", [])]
    except Exception as e:
        print(f"    [pexels] {e}")
        return []


def fetch_pixabay(query: str, api_key: str, per_page: int = 40, page: int = 1) -> List[str]:
    if not api_key:
        return []
    url = "https://pixabay.com/api/"
    params = {"key": api_key, "q": query, "per_page": per_page, "page": page,
              "image_type": "photo", "category": "people", "safesearch": "true"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return [h["webformatURL"] for h in resp.json().get("hits", [])]
    except Exception as e:
        print(f"    [pixabay] {e}")
        return []


def fetch_unsplash(query: str, api_key: str, per_page: int = 30, page: int = 1) -> List[str]:
    if not api_key:
        return []
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {api_key}"}
    params = {"query": query, "per_page": per_page, "page": page}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        return [r["urls"]["regular"] for r in resp.json().get("results", [])]
    except Exception as e:
        print(f"    [unsplash] {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def download_group(
    group: str, queries: List[str], target: int, out_dir: Path, size: int,
    clip_threshold: float, pexels_key: str, pixabay_key: str, unsplash_key: str,
) -> int:
    seen_hashes: Set[str] = set()
    saved = 0
    rejected = 0
    group_dir = out_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)

    existing = list(group_dir.glob("*.png")) + list(group_dir.glob("*.jpg"))
    if len(existing) >= target:
        print(f"  [{group}] Already have {len(existing)} >= {target}, skipping.")
        return len(existing)

    active = [s for s, k in [("Pexels", pexels_key), ("Pixabay", pixabay_key), ("Unsplash", unsplash_key)] if k]
    if not active:
        print(f"  [{group}] No API keys!")
        return 0

    print(f"  [{group}] Target: {target} | Sources: {', '.join(active)} | CLIP filter ON")

    for qi, query in enumerate(queries):
        if saved >= target:
            break
        print(f"    Query {qi+1}/{len(queries)}: \"{query}\"")

        all_urls: List[str] = []
        if pexels_key:
            urls = fetch_pexels(query, pexels_key, per_page=40)
            all_urls.extend(urls)
        if pixabay_key:
            urls = fetch_pixabay(query, pixabay_key, per_page=40)
            all_urls.extend(urls)
        if unsplash_key:
            urls = fetch_unsplash(query, unsplash_key, per_page=30)
            all_urls.extend(urls)

        print(f"      Candidates: {len(all_urls)}")

        for url in all_urls:
            if saved >= target:
                break
            img = _download_image(url)
            if img is None:
                continue
            if min(img.size) < 400:
                continue
            h = _content_hash(img)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            # CLIP verification
            passes, prob = clip_check(img, group, clip_threshold)
            if not passes:
                rejected += 1
                continue

            _save_image(img, out_dir, group, saved, size)
            saved += 1
            if saved % 5 == 0:
                print(f"      ✓ {saved}/{target} saved (rejected {rejected} mismatches)")

        time.sleep(0.3)

    print(f"  [{group}] DONE: {saved} saved, {rejected} rejected by CLIP")
    return saved


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Real Sample Downloader (with CLIP Filtering)")
    print("=" * 60)

    if not (args.pexels_key or args.pixabay_key or args.unsplash_key):
        print(
            "\n⚠️  No API keys! Get free keys:\n"
            "   Pexels:   https://www.pexels.com/api/\n"
            "   Pixabay:  https://pixabay.com/api/docs/\n"
            "   Unsplash: https://unsplash.com/developers\n"
        )
        return

    print(f"Output: {args.output_dir}")
    print(f"Per group: {args.per_group}")
    print(f"CLIP threshold: {args.clip_threshold}")
    print()

    total = 0
    for group, queries in SEARCH_QUERIES.items():
        total += download_group(
            group=group, queries=queries, target=args.per_group,
            out_dir=args.output_dir, size=args.size,
            clip_threshold=args.clip_threshold,
            pexels_key=args.pexels_key, pixabay_key=args.pixabay_key,
            unsplash_key=args.unsplash_key,
        )
        print()

    print("=" * 60)
    print(f"Total downloaded: {total}")
    print(f"Location: {args.output_dir}")


if __name__ == "__main__":
    main()
