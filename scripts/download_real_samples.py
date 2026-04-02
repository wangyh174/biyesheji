import os
import csv
import argparse
import requests
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import hashlib
import random


def calculate_phash(img_path):
    try:
        img = Image.open(img_path).convert("L").resize((8, 8), Image.Resampling.LANCZOS)
        pixels = np.array(img).flatten().tolist()
        avg = sum(pixels) / len(pixels)
        return "".join(["1" if p > avg else "0" for p in pixels])
    except Exception:
        return hashlib.md5(img_path.encode()).hexdigest()


class RealImageDownloader:
    def __init__(
        self,
        pexels_key=None,
        pixabay_key=None,
        unsplash_key=None,
        device="cpu",
    ):
        self.pexels_key = pexels_key
        self.pixabay_key = pixabay_key
        self.unsplash_key = unsplash_key
        self.device = device

        print(f"Loading CLIP model for STRICT disambiguation on {device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model.eval()

    def clip_probs(self, image, texts):
        inputs = self.processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
        return probs[0].detach().cpu().numpy().astype(float)

    def is_real_human_photo(self, image):
        try:
            texts = [
                "a real photo of a single human person",
                "a toy doll or figurine",
                "a cartoon or illustration",
                "an object product photo",
                "a group photo with many people",
            ]
            probs = self.clip_probs(image, texts)
            human_prob = float(probs[0])
            non_human_best = float(np.max(probs[1:]))
            return human_prob >= 0.50 and human_prob > non_human_best, human_prob
        except Exception:
            return False, 0.0

    def download_image(self, url, save_path):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(r.content)
                return True
        except Exception:
            pass
        return False

    def verify_image_disambiguated(self, img_path, group_name):
        try:
            image = Image.open(img_path).convert("RGB")
            gender = "male" if "male" in group_name else "female"
            other_gender = "female" if gender == "male" else "male"
            profession = "doctor" if "doctor" in group_name else "nurse"
            other_profession = "nurse" if profession == "doctor" else "doctor"

            is_human, human_prob = self.is_real_human_photo(image)
            if not is_human:
                return {
                    "passed": False,
                    "final_score": 0.0,
                    "human_prob": float(human_prob),
                    "target_prob": 0.0,
                    "competitor_best": 1.0,
                    "negative_best": 1.0,
                }

            target_text = (
                f"a real portrait photo of a single {gender} {profession} in a hospital, "
                f"medical uniform, clinical setting, healthcare worker"
            )

            competitor_texts = [
                f"a real portrait photo of a single {gender} {other_profession} in a hospital, healthcare worker",
                f"a real portrait photo of a single {other_gender} {profession} in a hospital, healthcare worker",
                f"a real portrait photo of a single {other_gender} {other_profession} in a hospital, healthcare worker",
            ]

            negative_texts = [
                "a toy doll or figurine",
                "a cartoon or illustration",
                "a mannequin or statue",
                "a normal portrait of a person, not a doctor or nurse",
                "a casual lifestyle portrait of one person",
                "an office worker or business person portrait",
                "a group photo with multiple people",
            ]

            texts = [target_text] + competitor_texts + negative_texts
            probs = self.clip_probs(image, texts)

            target_prob = float(probs[0])
            competitor_best = float(np.max(probs[1:4]))
            negative_best = float(np.max(probs[4:]))
            final_score = min(target_prob, human_prob)

            passed = (
                target_prob >= 0.40
                and target_prob > competitor_best + 0.05
                and target_prob > negative_best + 0.08
            )

            return {
                "passed": bool(passed),
                "final_score": float(final_score),
                "human_prob": float(human_prob),
                "target_prob": float(target_prob),
                "competitor_best": float(competitor_best),
                "negative_best": float(negative_best),
            }
        except Exception:
            return {
                "passed": False,
                "final_score": 0.0,
                "human_prob": 0.0,
                "target_prob": 0.0,
                "competitor_best": 1.0,
                "negative_best": 1.0,
            }

    def search_pexels(self, query, count=200):
        if not self.pexels_key:
            return []
        headers = {"Authorization": self.pexels_key}
        images = []
        for page in range(1, (count // 80) + 2):
            url = f"https://api.pexels.com/v1/search?query={query}&per_page=80&page={page}&orientation=portrait"
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code != 200:
                    break
                data = r.json()
                images.extend([{"url": p["src"]["large"], "source": "pexels"} for p in data.get("photos", [])])
                if len(images) >= count:
                    break
            except Exception:
                break
        return images[:count]

    def search_pixabay(self, query, count=500):
        if not self.pixabay_key:
            return []
        images = []
        for page in range(1, (count // 200) + 2):
            url = (
                f"https://pixabay.com/api/?key={self.pixabay_key}"
                f"&q={query.replace(' ', '+')}"
                f"&image_type=photo&safesearch=true&editors_choice=true"
                f"&per_page=200&page={page}"
            )
            try:
                r = requests.get(url, timeout=15)
                data = r.json()
                images.extend([{"url": p["webformatURL"], "source": "pixabay"} for p in data.get("hits", [])])
                if len(images) >= count:
                    break
            except Exception:
                break
        return images[:count]

    def search_unsplash(self, query, count=200):
        if not self.unsplash_key:
            return []
        headers = {"Authorization": f"Client-ID {self.unsplash_key}"}
        images = []
        per_page = 30
        for page in range(1, (count // per_page) + 2):
            url = (
                "https://api.unsplash.com/search/photos"
                f"?query={query.replace(' ', '%20')}"
                f"&page={page}&per_page={per_page}"
                "&orientation=portrait"
                "&content_filter=high"
            )
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code != 200:
                    break
                data = r.json()
                for p in data.get("results", []):
                    img_url = p.get("urls", {}).get("regular") or p.get("urls", {}).get("small")
                    if img_url:
                        images.append({"url": img_url, "source": "unsplash"})
                if len(images) >= count:
                    break
            except Exception:
                break
        return images[:count]


    def build_query_pool(self, group_name):
        query_map = {
            "male-doctor": [
                "male doctor white coat stethoscope hospital portrait",
                "male physician stethoscope clinical professional headshot",
                "man doctor medical uniform hospital",
                "male doctor scrubs clinic professional portrait",
                "male healthcare doctor uniform stethoscope",
            ],
            "female-doctor": [
                "female doctor white coat stethoscope hospital portrait",
                "female physician stethoscope clinical professional headshot",
                "woman doctor medical uniform hospital",
                "female doctor scrubs clinic professional portrait",
                "female healthcare doctor uniform stethoscope",
            ],
            "male-nurse": [
                "male nurse scrubs hospital professional portrait",
                "male nurse uniform stethoscope clinic headshot",
                "man nurse medical scrubs healthcare portrait",
                "male registered nurse hospital uniform professional",
                "male nurse blue scrubs clinical setting",
            ],
            "female-nurse": [
                "female nurse scrubs hospital professional portrait",
                "female nurse uniform stethoscope clinic headshot",
                "woman nurse medical scrubs healthcare portrait",
                "female registered nurse hospital uniform professional",
                "female nurse blue scrubs clinical setting",
            ],
        }
        return query_map.get(group_name, [group_name.replace("-", " ")])

    def fetch_group(self, group_name, output_dir, target_count=100, clip_threshold=0.22):
        print(f"\n--- Gathering REAL images (Deep Search 1000): {group_name} ---")
        os.makedirs(output_dir, exist_ok=True)
        candidate_dir = os.path.join(output_dir, "_candidates")
        os.makedirs(candidate_dir, exist_ok=True)
        candidate_csv = os.path.join(output_dir, "_candidate_scores.csv")

        downloaded_count = 0
        existing_hashes = set()
        seen_urls = set()
        candidate_rows = []

        for query_idx, query in enumerate(self.build_query_pool(group_name), start=1):
            if downloaded_count >= target_count:
                break

            print(f"Deep searching query {query_idx} for {group_name}: {query}")

            providers = [
                ("Unsplash", lambda: self.search_unsplash(query, 120)),
                ("Pexels", lambda: self.search_pexels(query, 200)),
                ("Pixabay", lambda: self.search_pixabay(query, 150)),
            ]
            
            query_threshold = clip_threshold if query_idx == 1 else max(0.20, clip_threshold - 0.01)

            for provider_name, provider_func in providers:
                if downloaded_count >= target_count:
                    break
                    
                print(f"  -> Pulling from {provider_name}...")
                provider_imgs = provider_func()
                random.shuffle(provider_imgs)
                
                potential_images = [img for img in provider_imgs if img["url"] not in seen_urls]
                for img in potential_images:
                    seen_urls.add(img["url"])
                    
                if not potential_images:
                    continue

                for p_img in tqdm(potential_images, desc=f"Evaluating {provider_name}"):
                    if downloaded_count >= target_count:
                        break

                    img_id = str(hashlib.md5(p_img["url"].encode()).hexdigest())[:10]
                    save_path = os.path.join(output_dir, f"{p_img['source']}_{img_id}.jpg")

                    if self.download_image(p_img["url"], save_path):
                        result = self.verify_image_disambiguated(save_path, group_name)
                        h = calculate_phash(save_path)

                        candidate_rows.append(
                            {
                                "group": group_name,
                                "query_idx": query_idx,
                                "query": query,
                                "source": p_img["source"],
                                "url": p_img["url"],
                                "file_name": os.path.basename(save_path),
                                "human_prob": result["human_prob"],
                                "target_prob": result["target_prob"],
                                "competitor_best": result["competitor_best"],
                                "negative_best": result["negative_best"],
                                "final_score": result["final_score"],
                                "passed": int(result["passed"]),
                            }
                        )

                        if result["passed"] and result["final_score"] >= query_threshold and h not in existing_hashes:
                            existing_hashes.add(h)
                            downloaded_count += 1
                        else:
                            candidate_path = os.path.join(candidate_dir, os.path.basename(save_path))
                            if os.path.exists(candidate_path):
                                os.remove(candidate_path)
                            os.replace(save_path, candidate_path)

        candidate_rows.sort(key=lambda x: x["final_score"], reverse=True)
        with open(candidate_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "group",
                    "query_idx",
                    "query",
                    "source",
                    "url",
                    "file_name",
                    "human_prob",
                    "target_prob",
                    "competitor_best",
                    "negative_best",
                    "final_score",
                    "passed",
                ],
            )
            writer.writeheader()
            writer.writerows(candidate_rows)

        print(f"Final Count for {group_name}: {downloaded_count}/{target_count}")
        print(f"Saved candidate review file: {candidate_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pexels-key", type=str, default="VriU36HKC2dGu63uX8Uoseqf6wwvsECXoas33RoiGFSO5ItcIwsYAKfq")
    parser.add_argument("--pixabay-key", type=str, default="55245278-eb83bc54c887305bb0c422185")
    parser.add_argument("--unsplash-key", type=str, default="fsqS67ZO7mOGFKto3dbAm-JQOLMVT6I3E7qMh7J6lHU")
    parser.add_argument("--samples-per-group", type=int, default=100)
    parser.add_argument("--clip-threshold", type=float, default=0.22)
    args = parser.parse_args()

    downloader = RealImageDownloader(
        pexels_key=args.pexels_key,
        pixabay_key=args.pixabay_key,
        unsplash_key=args.unsplash_key,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    for group in ["male-doctor", "female-doctor", "male-nurse", "female-nurse"]:
        downloader.fetch_group(
            group,
            os.path.join("data/real_samples", group),
            args.samples_per_group,
            args.clip_threshold,
        )


if __name__ == "__main__":
    main()
