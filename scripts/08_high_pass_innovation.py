import os
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from glob import glob

def apply_high_pass_filter(img):
    """
    Isolates High-frequency noise (Residuals) while suppressing low-frequency semantics.
    Reference: Innovation 1 - High-pass Residual Decoupling.
    """
    # 1. Convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Laplacian filter to get high-frequency residuals
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    
    # 3. Normalize to [0, 255] for visual consistency (though some detectors might work better with raw)
    normalized = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)
    res = np.uint8(normalized)
    
    # 4. Optional: Combine back into 3 channels if detector requires it
    res_bgr = cv2.cvtColor(res, cv2.COLOR_GRAY2BGR)
    
    return res_bgr

def main():
    parser = argparse.ArgumentParser(description="High-pass Residual Decoupling Innovation (Project Innovation 1)")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--process-all", action="store_true", help="Process all data or just a sample")
    args = parser.parse_args()

    data_dir = os.path.join(args.project_root, "data")
    output_dir = os.path.join(data_dir, "high_pass_residuals")
    os.makedirs(output_dir, exist_ok=True)

    sources = [
        {"path": os.path.join(data_dir, "generated_raw"), "label": "fake"},
        {"path": os.path.join(data_dir, "real_samples"), "label": "real"}
    ]

    print(f"--- Stage 8: Semantic-Noise Decoupling (High-pass Residuals) ---")
    
    for source in sources:
        if not os.path.exists(source["path"]): continue
        groups = [d for d in os.listdir(source["path"]) if os.path.isdir(os.path.join(source["path"], d))]
        
        for group in groups:
            group_in = os.path.join(source["path"], group)
            group_out = os.path.join(output_dir, source["label"], group)
            os.makedirs(group_out, exist_ok=True)
            
            images = glob(os.path.join(group_in, "*.png")) + glob(os.path.join(group_in, "*.jpg"))
            if not args.process_all:
                images = images[:30] # Sample mode
                
            print(f"Filtering {len(images)} images in {source['label']}/{group}...")
            
            for img_path in tqdm(images):
                img = cv2.imread(img_path)
                if img is None: continue
                # Basic resize
                img = cv2.resize(img, (256, 256))
                
                hpf_img = apply_high_pass_filter(img)
                out_name = os.path.basename(img_path)
                cv2.imwrite(os.path.join(group_out, out_name), hpf_img)

    print(f"Done. Residual images saved to {output_dir}")
    print("Next step: Run existing detectors (scripts/03) on this high_pass_residuals directory.")

if __name__ == "__main__":
    main()
