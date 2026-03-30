"""
Download real photographs from multiple free stock image APIs
for use as the 'Real' control group in fairness evaluation.

Sources (all free, no watermarks, commercial-use-OK):
  1. Pexels  (https://www.pexels.com)
  2. Pixabay (https://pixabay.com)
  3. Unsplash (https://unsplash.com)

Usage:
  # Run on local machine (needs internet):
  python scripts/download_real_samples.py --per-group 60 --output-dir data/real_samples

  # Run on Colab:
  !python scripts/download_real_samples.py --per-group 60

API keys are FREE to obtain:
  Pexels:   https://www.pexels.com/api/  (instant, no review)
  Pixabay:  https://pixabay.com/api/docs/ (instant, no review)
  Unsplash: https://unsplash.com/developers (instant, no review)

You can set them as environment variables or pass them as arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import quote_plus

import requests
from PIL import Image


# ─── search query templates per group ───────────────────────────────────────
# Each group has multiple diverse queries to maximize variety
SEARCH_QUERIES: Dict[str, List[str]] = {
    "male-doctor": [
        "male doctor portrait hospital",
        "man doctor stethoscope",
        "male physician clinic",
        "man surgeon hospital",
        "male doctor white coat",
        "man medical doctor office",
        "male healthcare doctor",
        "young man doctor hospital",
        "male doctor patient consultation",
        "man doctor medical uniform",
    ],
    "female-doctor": [
        "female doctor portrait hospital",
        "woman doctor stethoscope",
        "female physician clinic",
        "woman surgeon hospital",
        "female doctor white coat",
        "woman medical doctor office",
        "female healthcare doctor",
        "young woman doctor hospital",
        "female doctor patient consultation",
        "woman doctor medical uniform",
    ],
    "male-nurse": [
        "male nurse portrait hospital",
        "man nurse scrubs",
        "male nurse caring patient",
        "man nursing hospital ward",
        "male nurse medical uniform",
        "man registered nurse",
        "male nurse healthcare",
        "young man nurse hospital",
        "male nurse clinic",
        "man nurse patient care",
    ],
    "female-nurse": [
        "female nurse portrait hospital",
        "woman nurse scrubs",
        "female nurse caring patient",
        "woman nursing hospital ward",
        "female nurse medical uniform",
        "woman registered nurse",
        "female nurse healthcare",
        "young woman nurse hospital",
        "female nurse clinic",
        "woman nurse patient care",
    ],
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Download real stock photos for fairness evaluation"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "real_samples",
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=60,
        help="Target number of images per group (will try to get at least this many)",
    )
    parser.add_argument("--size", type=int, default=512, help="Resize to NxN")
    parser.add_argument(
        "--pexels-key",
        type=str,
        default=os.environ.get("PEXELS_API_KEY", ""),
        help="Pexels API key (or set PEXELS_API_KEY env var)",
    )
    parser.add_argument(
        "--pixabay-key",
        type=str,
        default=os.environ.get("PIXABAY_API_KEY", ""),
        help="Pixabay API key (or set PIXABAY_API_KEY env var)",
    )
    parser.add_argument(
        "--unsplash-key",
        type=str,
        default=os.environ.get("UNSPLASH_ACCESS_KEY", ""),
        help="Unsplash Access Key (or set UNSPLASH_ACCESS_KEY env var)",
    )
    return parser.parse_args()


def _download_image(url: str, timeout: int = 15) -> Optional[Image.Image]:
    """Download an image from a URL and return as PIL Image."""
    try:
        resp = requests.get(url, timeout=timeout, stream=True)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        print(f"    [warn] download failed: {e}")
        return None


def _content_hash(img: Image.Image) -> str:
    """Quick perceptual hash to avoid saving duplicate images."""
    thumb = img.resize((16, 16)).convert("L")
    raw = thumb.tobytes()
    return hashlib.md5(raw).hexdigest()


def _save_image(
    img: Image.Image,
    out_dir: Path,
    group: str,
    idx: int,
    size: int,
) -> Path:
    """Resize, center-crop, and save."""
    # Center crop to square
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
#  Source 1: Pexels API
# ═══════════════════════════════════════════════════════════════════════════
def fetch_pexels(
    query: str, api_key: str, per_page: int = 40, page: int = 1
) -> List[str]:
    """Return list of image URLs from Pexels search."""
    if not api_key:
        return []
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": api_key}
    params = {"query": query, "per_page": per_page, "page": page, "orientation": "square"}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [p["src"]["medium"] for p in data.get("photos", [])]
    except Exception as e:
        print(f"    [pexels] error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  Source 2: Pixabay API
# ═══════════════════════════════════════════════════════════════════════════
def fetch_pixabay(
    query: str, api_key: str, per_page: int = 40, page: int = 1
) -> List[str]:
    """Return list of image URLs from Pixabay search."""
    if not api_key:
        return []
    url = "https://pixabay.com/api/"
    params = {
        "key": api_key,
        "q": query,
        "per_page": per_page,
        "page": page,
        "image_type": "photo",
        "category": "people",
        "safesearch": "true",
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [h["webformatURL"] for h in data.get("hits", [])]
    except Exception as e:
        print(f"    [pixabay] error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  Source 3: Unsplash API
# ═══════════════════════════════════════════════════════════════════════════
def fetch_unsplash(
    query: str, api_key: str, per_page: int = 30, page: int = 1
) -> List[str]:
    """Return list of image URLs from Unsplash search."""
    if not api_key:
        return []
    url = "https://api.unsplash.com/search/photos"
    headers = {"Authorization": f"Client-ID {api_key}"}
    params = {"query": query, "per_page": per_page, "page": page}
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [r["urls"]["regular"] for r in data.get("results", [])]
    except Exception as e:
        print(f"    [unsplash] error: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════
def download_group(
    group: str,
    queries: List[str],
    target: int,
    out_dir: Path,
    size: int,
    pexels_key: str,
    pixabay_key: str,
    unsplash_key: str,
) -> int:
    """Download images for one group from all sources."""
    seen_hashes: Set[str] = set()
    saved = 0
    group_dir = out_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)

    # Check how many we already have
    existing = list(group_dir.glob("*.png")) + list(group_dir.glob("*.jpg"))
    if len(existing) >= target:
        print(f"  [{group}] Already have {len(existing)} images >= target {target}, skipping.")
        return len(existing)

    active_sources = []
    if pexels_key:
        active_sources.append("Pexels")
    if pixabay_key:
        active_sources.append("Pixabay")
    if unsplash_key:
        active_sources.append("Unsplash")

    if not active_sources:
        print(f"  [{group}] ERROR: No API keys provided! Cannot download.")
        return 0

    print(f"  [{group}] Target: {target} images | Sources: {', '.join(active_sources)}")

    for qi, query in enumerate(queries):
        if saved >= target:
            break

        print(f"    Query {qi+1}/{len(queries)}: \"{query}\"")

        # Collect URLs from all sources
        all_urls: List[str] = []

        if pexels_key:
            urls = fetch_pexels(query, pexels_key, per_page=30)
            print(f"      Pexels: {len(urls)} results")
            all_urls.extend(urls)

        if pixabay_key:
            urls = fetch_pixabay(query, pixabay_key, per_page=30)
            print(f"      Pixabay: {len(urls)} results")
            all_urls.extend(urls)

        if unsplash_key:
            urls = fetch_unsplash(query, unsplash_key, per_page=30)
            print(f"      Unsplash: {len(urls)} results")
            all_urls.extend(urls)

        # Download and deduplicate
        for url in all_urls:
            if saved >= target:
                break
            img = _download_image(url)
            if img is None:
                continue
            # Check minimum resolution
            if min(img.size) < 256:
                continue
            h = _content_hash(img)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            path = _save_image(img, out_dir, group, saved, size)
            saved += 1
            if saved % 10 == 0:
                print(f"      => {saved}/{target} saved")

        # Respect rate limits
        time.sleep(0.5)

    print(f"  [{group}] DONE: {saved} images saved to {group_dir}")
    return saved


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("Real Sample Downloader for Fairness Evaluation")
    print("=" * 60)

    has_any_key = bool(args.pexels_key or args.pixabay_key or args.unsplash_key)
    if not has_any_key:
        print(
            "\n⚠️  No API keys detected! You need at least one.\n"
            "   Get FREE keys (instant, no review):\n"
            "   • Pexels:   https://www.pexels.com/api/\n"
            "   • Pixabay:  https://pixabay.com/api/docs/\n"
            "   • Unsplash: https://unsplash.com/developers\n"
            "\n   Then run:\n"
            '   python scripts/download_real_samples.py --pexels-key "YOUR_KEY" --pixabay-key "YOUR_KEY"\n'
            "   or set environment variables: PEXELS_API_KEY, PIXABAY_API_KEY, UNSPLASH_ACCESS_KEY\n"
        )
        return

    print(f"Output directory: {args.output_dir}")
    print(f"Target per group: {args.per_group}")
    print(f"Image size: {args.size}x{args.size}")
    print()

    total = 0
    for group, queries in SEARCH_QUERIES.items():
        total += download_group(
            group=group,
            queries=queries,
            target=args.per_group,
            out_dir=args.output_dir,
            size=args.size,
            pexels_key=args.pexels_key,
            pixabay_key=args.pixabay_key,
            unsplash_key=args.unsplash_key,
        )
        print()

    print("=" * 60)
    print(f"All done! Total images downloaded: {total}")
    print(f"Location: {args.output_dir}")
    print(
        "\nNext step: re-run the detection pipeline with these real samples:\n"
        "  python scripts/00_run_local_pipeline.py \\\n"
        "    --project-root ./ \\\n"
        "    --generator fairdiffusion \\\n"
        "    --samples-per-group 40 \\\n"
        "    --real-per-group 40 \\\n"
        "    --real-source local \\\n"
        "    --detectors cnndetection,f3net,lgrad \\\n"
        "    --clip-min-score 0.22"
    )


if __name__ == "__main__":
    main()
