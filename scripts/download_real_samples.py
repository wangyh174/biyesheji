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
        print(f"Loading CLIP model for semantic verification on {device}...")
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

    def verify_image(self, img_path, target_prompt):
        """Verify if image matches target (e.g. 'a male doctor') with CLIP"""
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(text=[target_prompt, "unrelated background", "medical equipment only"], 
                                    images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            score = probs[0][0].item()
            return score
        except:
            return 0.0

    def search_pexels(self, query, count=50):
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

    def search_pixabay(self, query, count=50):
        if not self.pixabay_key: return []
        url = f"https://pixabay.com/api/?key={self.pixabay_key}&q={query.replace(' ', '+')}&image_type=photo&per_page={count}"
        try:
            r = requests.get(url)
            r.raise_for_status()
            data = r.json()
            return [{"url": p["webformatURL"], "source": "pixabay"} for p in data.get("hits", [])]
        except:
            return []

    def fetch_group(self, group_name, output_dir, target_count=50, clip_threshold=0.22):
        print(f"\n--- Gathering REAL images for group: {group_name} ---")
        os.makedirs(output_dir, exist_ok=True)
        
        # Mapping group names to queries
        query_map = {
            "male-doctor": "male doctor, man doctor professional",
            "female-doctor": "female doctor, woman doctor hospital",
            "male-nurse": "male nurse, man nurse practitioner",
            "female-nurse": "female nurse, woman medical nurse"
        }
        
        query = query_map.get(group_name, group_name.replace('-', ' '))
        clip_prompt = f"a photo of a {group_name.replace('-', ' ')} in professional medical setting"
        
        # Search both libraries
        potential_images = self.search_pexels(query, target_count*2) + self.search_pixabay(query, target_count*2)
        
        downloaded_count = 0
        existing_hashes = set()
        
        for p_img in tqdm(potential_images):
            if downloaded_count >= target_count: break
            
            img_id = p_img["url"].split('/')[-1].split('?')[0]
            save_path = os.path.join(output_dir, f"{p_img['source']}_{img_id}")
            if not save_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                save_path += ".jpg"

            if self.download_image(p_img["url"], save_path):
                # CLIP Check
                score = self.verify_image(save_path, clip_prompt)
                
                # Check Deduplication (pHash)
                h = calculate_phash(save_path)
                
                if score >= clip_threshold and h not in existing_hashes:
                    existing_hashes.add(h)
                    downloaded_count += 1
                else:
                    # Remove low quality or duplicate
                    if os.path.exists(save_path): os.remove(save_path)
        
        print(f"Successfully collected {downloaded_count} verified images for {group_name}.")

def main():
    parser = argparse.ArgumentParser(description="Real Image Data Gatherer (CLIP Synchronized)")
    parser.add_argument("--pexels-key", type=str, default="563492ad6f917000010000018f6f368097b6452296d11a6873523fe9", help="Pexels API Key")
    parser.add_argument("--pixabay-key", type=str, default="55245278-eb83bc54c887305bb0c422185", help="Pixabay API Key")
    parser.add_argument("--output-dir", type=str, default="data/real_samples")
    parser.add_argument("--samples-per-group", type=int, default=50)
    parser.add_argument("--clip-threshold", type=float, default=0.22)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    downloader = RealImageDownloader(args.pexels_key, args.pixabay_key, device)

    groups = ["male-doctor", "female-doctor", "male-nurse", "female-nurse"]
    
    for group in groups:
        downloader.fetch_group(group, os.path.join(args.output_dir, group), args.samples_per_group, args.clip_threshold)

if __name__ == "__main__":
    main()
