import os
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from glob import glob

def patch_shuffle(img, patch_size_n=8):
    """
    Splits image into n x n patches and shuffles them randomly.
    Img: 512x512, n=8 -> 64x64 patches.
    """
    h, w, c = img.shape
    patch_h, patch_w = h // patch_size_n, w // patch_size_n
    
    patches = []
    for i in range(patch_size_n):
        for j in range(patch_size_n):
            patch = img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w, :]
            patches.append(patch)
            
    np.random.shuffle(patches)
    
    shuffled_img = np.zeros_like(img)
    idx = 0
    for i in range(patch_size_n):
        for j in range(patch_size_n):
            shuffled_img[i*patch_h:(i+1)*patch_h, j*patch_w:(j+1)*patch_w, :] = patches[idx]
            idx += 1
            
    return shuffled_img

def main():
    parser = argparse.ArgumentParser(description="Patch Shuffling (Structural Attribution) - Lin CVPR 24 & BSA NeurIPS 24")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--patch-n", type=int, default=8, help="Number of patches per side (e.g. 8x8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling consistency")
    args = parser.parse_args()

    np.random.seed(args.seed)
    
    data_dir = os.path.join(args.project_root, "data")
    output_dir = os.path.join(data_dir, f"shuffled_{args.patch_n}x{args.patch_n}")
    os.makedirs(output_dir, exist_ok=True)

    # Process both Fake (Generated) and Real (Real-Samples)
    sources = [
        {"path": os.path.join(data_dir, "generated_raw"), "label": "fake"},
        {"path": os.path.join(data_dir, "real_samples"), "label": "real"}
    ]

    print(f"--- Stage 6: Structural Attribution (Patch Shuffling {args.patch_n}x{args.patch_n}) ---")
    
    manifest = []
    
    for source in sources:
        # We need to traverse demographic groups
        if not os.path.exists(source["path"]):
            continue
            
        groups = [d for d in os.listdir(source["path"]) if os.path.isdir(os.path.join(source["path"], d))]
        
        for group in groups:
            group_in = os.path.join(source["path"], group)
            group_out = os.path.join(output_dir, source["label"], group)
            os.makedirs(group_out, exist_ok=True)
            
            images = glob(os.path.join(group_in, "*.png")) + glob(os.path.join(group_in, "*.jpg"))
            print(f"Shuffling {len(images)} images in {source['label']}/{group}...")
            
            for img_path in tqdm(images):
                img = cv2.imread(img_path)
                if img is None: continue
                # Resize to 512x512 if not already (for consistent patch size logic)
                if img.shape[0] != 512 or img.shape[1] != 512:
                    img = cv2.resize(img, (512, 512))
                
                shuffled = patch_shuffle(img, args.patch_n)
                out_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(group_out, out_name), shuffled)
                
                manifest.append({
                    "original_path": img_path,
                    "shuffled_path": os.path.join(group_out, out_name),
                    "label": source["label"],
                    "group": group,
                    "patch_n": args.patch_n
                })

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(os.path.join(output_dir, "manifest.csv"), index=False)
    print(f"Done. Shuffled images saved to {output_dir}")

if __name__ == "__main__":
    main()
