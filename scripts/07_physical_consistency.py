import os
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
from glob import glob
from skimage.feature import graycomatrix, graycoprops

def extract_glcm_features(img, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256):
    """
    Extracts Second-Order Features (GLCM) - Reference: Zheng-D3 ICCV 2025
    Focusing on pixel-to-pixel co-occurrence rather than semantic contents.
    """
    # Convert to grayscale first
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Compute the GLCM matrix
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)
    
    # Extract common properties
    features = {
        "contrast": np.mean(graycoprops(glcm, 'contrast')),
        "dissimilarity": np.mean(graycoprops(glcm, 'dissimilarity')),
        "homogeneity": np.mean(graycoprops(glcm, 'homogeneity')),
        "energy": np.mean(graycoprops(glcm, 'energy')),
        "correlation": np.mean(graycoprops(glcm, 'correlation')),
        "ASM": np.mean(graycoprops(glcm, 'ASM'))
    }
    
    return features

def main():
    parser = argparse.ArgumentParser(description="Second-Order Physical Feature Analysis (D3 ICCV 2025)")
    parser.add_argument("--project-root", type=str, default=".", help="Project root directory")
    parser.add_argument("--max-samples", type=int, default=50, help="Max samples per group to keep analysis quick")
    args = parser.parse_args()

    data_root = os.path.join(args.project_root, "data")
    
    # Source paths: generated_raw (AI) and real_samples (Human)
    sources = [
        {"path": os.path.join(data_root, "generated_raw"), "label": "fake"},
        {"path": os.path.join(data_root, "real_samples"), "label": "real"}
    ]
    
    all_results = []
    
    print(f"--- Stage 7: Physical Consistency Analysis (Zheng-D3 ICCV 2025) ---")
    
    for source in sources:
        if not os.path.exists(source["path"]): continue
        
        groups = [d for d in os.listdir(source["path"]) if os.path.isdir(os.path.join(source["path"], d))]
        if source["label"] == "real":
            groups = [d for d in groups if d.endswith("_after")]
        
        for group in groups:
            group_dir = os.path.join(source["path"], group)
            images = glob(os.path.join(group_dir, "*.png")) + glob(os.path.join(group_dir, "*.jpg"))
            
            # Sub-sample if needed
            selected = images[:args.max_samples]
            print(f"Extracting GLCM for {len(selected)} samples in {source['label']}/{group}...")
            
            for img_path in tqdm(selected):
                img = cv2.imread(img_path)
                if img is None: continue
                # Resize for consistent stats
                img = cv2.resize(img, (256, 256))
                
                feats = extract_glcm_features(img)
                feats["group"] = group
                feats["label"] = source["label"]
                feats["path"] = img_path
                all_results.append(feats)

    results_df = pd.DataFrame(all_results)
    
    # Aggregate and show
    summary = results_df.groupby(["label", "group"]).mean(numeric_only=True).reset_index()
    print("\n--- Physical Consistency (Mean Second-Order Features) ---")
    print(summary)
    
    output_csv = os.path.join(data_root, "physical_consistency_results.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"\nDetailed physical features saved to {output_csv}")

if __name__ == "__main__":
    main()
