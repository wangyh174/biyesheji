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
        print(f"Loading CLIP model for STRICT disambiguation on {device}...")
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
        try:
            image = Image.open(img_path).convert("RGB")
            gender = "male" if "male" in group_name else "female"
            
            target_text = f"a photo of a {group_name.replace('-', ' ')}"
            competitor_text = f"a photo of a {gender} doctor portrait" if "nurse" in group_name else f"a photo of a {gender} nurse portrait"
            
            inputs = self.processor(text=[target_text, competitor_text], 
                                    images=image, return_tensors="pt", padding=True).to(self.device)
            outputs = self.model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1)
            
            target_prob = probs[0][0].item()
            competitor_prob = probs[0][1].item()
            
            # --- STRICT WINNER LOGIC --- (User preference: Quality over Quantity)
            if target_prob > competitor_prob: # No soft margin
                return target_prob
            return 0.0
        except:
            return 0.0

    def search_pexels(self, query, count=500):
        if not self.pexels_key: return []
        headers = {"Authorization": self.pexels_key}
        images = []
        # Multi-page paging to reach high counts
        for page in range(1, (count // 80) + 2):
            url = f"https://api.pexels.com/v1/search?query={query}&per_page=80&page={page}"
            try:
                r = requests.get(url, headers=headers)
                data = r.json()
                images.extend([{"url": p["src"]["large"], "source": "pexels"} for p in data.get("photos", [])])
                if len(images) >= count: break
            except:
                break
        return images[:count]

    def search_pixabay(self, query, count=500):
        if not self.pixabay_key: return []
        # Pixabay max 200 per page, paging if needed
        images = []
        for page in range(1, (count // 200) + 2):
            url = f"https://pixabay.com/api/?key={self.pixabay_key}&q={query.replace(' ', '+')}&image_type=photo&per_page=200&page={page}"
            try:
                r = requests.get(url)
                data = r.json()
                images.extend([{"url": p["webformatURL"], "source": "pixabay"} for p in data.get("hits", [])])
                if len(images) >= count: break
            except:
                break
        return images[:count]

    def fetch_group(self, group_name, output_dir, target_count=50, clip_threshold=0.22):
        print(f"\n--- Gathering REAL images (Deep Search 1000): {group_name} ---")
        os.makedirs(output_dir, exist_ok=True)
        
        # Specific queries
        query_map = {
            "male-doctor": "male doctor hospital",
            "female-doctor": "female doctor hospital",
            "male-nurse": "male nurse uniform scrubs",
            "female-nurse": "female nurse clinic"
        }
        
        query = query_map.get(group_name, group_name.replace('-', ' '))
        print(f"Deep searching up to 1000 candidates for {group_name}...")
        potential_images = self.search_pexels(query, 500) + self.search_pixabay(query, 500)
        
        # Shuffle search results slightly for variety
        import random
        random.shuffle(potential_images)

        downloaded_count = 0
        existing_hashes = set()
        
        for p_img in tqdm(potential_images):
            if downloaded_count >= target_count: break
            img_id = str(hashlib.md5(p_img["url"].encode()).hexdigest())[:10]
            save_path = os.path.join(output_dir, f"{p_img['source']}_{img_id}.jpg")

            if self.download_image(p_img["url"], save_path):
                score = self.verify_image_disambiguated(save_path, group_name)
                h = calculate_phash(save_path)
                
                if score >= clip_threshold and h not in existing_hashes:
                    existing_hashes.add(h)
                    downloaded_count += 1
                else:
                    if os.path.exists(save_path): os.remove(save_path)
        
        print(f"Final Count for {group_name}: {downloaded_count}/{target_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pexels-key", type=str, default="563492ad6f917000010000018f6f368097b6452296d11a6873523fe9")
    parser.add_argument("--pixabay-key", type=str, default="55245278-eb83bc54c887305bb0c422185")
    parser.add_argument("--samples-per-group", type=int, default=50)
    parser.add_argument("--clip-threshold", type=float, default=0.22) # Restore strict threshold
    args = parser.parse_args()

    downloader = RealImageDownloader(args.pexels_key, args.pixabay_key, "cuda" if torch.cuda.is_available() else "cpu")
    # All groups strictly disambiguated
    for group in ["male-doctor", "female-doctor", "male-nurse", "female-nurse"]:
        downloader.fetch_group(group, os.path.join("data/real_samples", group), args.samples_per_group, args.clip_threshold)

if __name__ == "__main__":
    main()
