import os
import argparse
import requests
import torch
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import hashlib

def calculate_phash(img_path):
    """Simple perceptual hash for deduplication"""
    try:
        img = Image.open(img_path).convert('L').resize((8, 8), Image.Resampling.LANCZOS)
        pixels = list(img.getdata())
        avg = sum(pixels) / len(pixels)
        return "".join(['1' if p > avg else '0' for p in pixels])
    except:
        return hashlib.md5(img_path.encode()).hexdigest()

class RealImageDownloader:
    def __init__(self, pexels_key=None, pixabay_key=None, device="cpu"):
        self.pexels_key = pexels_key
        self.pixabay_key = pixabay_key
        self.device = device
        
        # Load CLIP for semantic verification
        print(f"Loading CLIP model for semantic disambiguation on {device}...")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def download_image(self, url, save_path):
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                with open(save_path, 'wb') as f:
                    f.write(r.content)
                return True
        except:
            pass
        return False

    def verify_image_disambiguated(self, img_path, group_name):
        """
        Verify image using competitive semantics.
        Ensures a 'Nurse' image is more 'Nurse-like' than 'Doctor-like'.
        """
        try:
            image = Image.open(img_path).convert("RGB")
            # Determine competitors based on gender
            gender = "male" if "male" in group_name else "female"
            
            # Semantic competitors
            target_text = f"a photo of a {group_name.replace('-', ' ')}"
            competitor_text = f"a photo of a {gender} doctor" if "nurse" in group_name else f"a photo of a {gender} nurse"
            background_text = "medical equipment or hospital corridor without people"
            
            inputs = self.processor(text=[target_text, competitor_text, background_text], 
                                    images=image, return_tensors="pt", padding=True).to(self.device)
            
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1) # [batch, 3]
            
            target_prob = probs[0][0].item()
            competitor_prob = probs[0][1].item()
            
            # Rule: Target must be the winner AND above threshold
            # If target_prob is lower than competitor_prob, reject (Score = 0)
            if target_prob > competitor_prob:
                return target_prob
            return 0.0
        except:
            return 0.0

    def search_pexels(self, query, count=100):
        if not self.pexels_key: return []
        headers = {"Authorization": self.pexels_key}
        url = f"https://api.pexels.com/v1/search?query={query}&per_page={count}"
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            data = r.json()
            return [{"url": p["src"]["large"], "source": "pexels"} for p in data.get("photos", [])]
        except:
            return []

    def search_pixabay(self, query, count=100):
        if not self.pixabay_key: return []
        url = f"https://pixabay.com/api/?key={self.pixabay_key}&q={query.replace(' ', '+')}&image_type=photo&per_page={count}"
        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
            return [{"url": p["webformatURL"], "source": "pixabay"} for p in data.get("hits", [])]
        except:
            return []

    def fetch_group(self, group_name, output_dir, target_count=100, clip_threshold=0.22):
        print(f"\n--- Gathering REAL images with Disambiguation: {group_name} ---")
        os.makedirs(output_dir, exist_ok=True)
        
        query_map = {
            "male-doctor": "male doctor professional",
            "female-doctor": "female doctor portrait",
            "male-nurse": "male nurse medical scrubs",
            "female-nurse": "female nurse nursing clinic"
        }
        
        query = query_map.get(group_name, group_name.replace('-', ' '))
        potential_images = self.search_pexels(query, target_count*4) + self.search_pixabay(query, target_count*4)
        
        downloaded_count = 0
        existing_hashes = set()
        
        for p_img in tqdm(potential_images):
            if downloaded_count >= target_count: break
            
            img_id = p_img["url"].split('/')[-1].split('?')[0]
            save_path = os.path.join(output_dir, f"{p_img['source']}_{img_id}")
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                save_path += ".jpg"

            if self.download_image(p_img["url"], save_path):
                # Advanced Disambiguation Check
                score = self.verify_image_disambiguated(save_path, group_name)
                
                # Deduplication
                h = calculate_phash(save_path)
                
                if score >= clip_threshold and h not in existing_hashes:
                    existing_hashes.add(h)
                    downloaded_count += 1
                else:
                    if os.path.exists(save_path): os.remove(save_path)
        
        print(f"Collected {downloaded_count} disambiguated images for {group_name}.")

def main():
    parser = argparse.ArgumentParser(description="Real Image Gatherer (Semantic Disambiguation Mode)")
    parser.add_argument("--pexels-key", type=str, default="563492ad6f917000010000018f6f368097b6452296d11a6873523fe9")
    parser.add_argument("--pixabay-key", type=str, default="55245278-eb83bc54c887305bb0c422185")
    parser.add_argument("--output-dir", type=str, default="data/real_samples")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    downloader = RealImageDownloader(args.pexels_key, args.pixabay_key, device)

    groups = ["male-doctor", "female-doctor", "male-nurse", "female-nurse"]
    for group in groups:
        downloader.fetch_group(group, os.path.join(args.output_dir, group), args.samples_per_group, args.clip_threshold)

if __name__ == "__main__":
    main()
