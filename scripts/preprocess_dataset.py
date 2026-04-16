# -*- coding: utf-8 -*-
import os
import cv2
import csv
import sys
from glob import glob
from mtcnn import MTCNN

# 强制控制台使用 UTF-8 输出，防止 print 乱码
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

print("正在加载 MTCNN 模型 (当前设定：单一固定比例 + 最大满充抢救)...")
detector = MTCNN()

def process_real_images_strict(input_dir, output_dir, metadata_list, group_name, target_size=512):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob(os.path.join(input_dir, '*.*'))
    success_count = 0
    fallback_count = 0 
    error_count = 0 
    
    discard_orig_small = 0   
    discard_need_upsample = 0 
    salvage_too_large = 0     # 记录被“最大满充”抢救的大头照数量

    # ==========================================
    # 🎯 唯一核心超参数：你可以修改这里！
    # 0.35 = 标准半身照 (推荐科研使用，极其严格)
    # 0.2  = 远景宽视野 (如果你实在缺数据，可以改回0.2)
    # ==========================================
    FACE_TARGET_RATIO = 0.3
    
    print(f"--> 正在处理组别: {group_name} | 候选总数: {len(image_paths)}")

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: 
            error_count += 1
            continue
        
        h_orig, w_orig = img.shape[:2]

        if h_orig < target_size or w_orig < target_size:
            discard_orig_small += 1
            continue

        scale_factor = 1.0
        MAX_DETECTION_SIZE = 1500  

        if max(h_orig, w_orig) > MAX_DETECTION_SIZE:
            scale_factor = MAX_DETECTION_SIZE / max(h_orig, w_orig)
            det_w = int(w_orig * scale_factor)
            det_h = int(h_orig * scale_factor)
            img_for_det = cv2.resize(img, (det_w, det_h), interpolation=cv2.INTER_AREA)
        else:
            img_for_det = img

        img_rgb = cv2.cvtColor(img_for_det, cv2.COLOR_BGR2RGB)
        
        try:
            faces = detector.detect_faces(img_rgb)
        except Exception:
            error_count += 1
            continue

        if faces:
            face = max(faces, key=lambda rect: rect['box'][2] * rect['box'][3])
            
            x, y, w, h = [int(v / scale_factor) for v in face['box']]
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # ---------------------------------------------------------
            # 铁腕逻辑：只认你设定的 FACE_TARGET_RATIO
            # ---------------------------------------------------------
            ideal_side = int(w / FACE_TARGET_RATIO)
            
            # 【红线】：如果为了达到该比例，框小于512，必须放大，坚决丢弃！
            if ideal_side < target_size:
                discard_need_upsample += 1
                continue
                
            # 【抢救】：如果算出的框比原图还大（说明原图是特写）
            if ideal_side > min(h_orig, w_orig):
                # 不改变比例妥协，直接切能切出的最大正方形
                dynamic_side = min(h_orig, w_orig)
                salvage_too_large += 1
            else:
                # 完美切出目标比例
                dynamic_side = ideal_side
            
            # 起刀点死死锁住 0.35 (黄金分割：让头顶上方留35%，下方留65%)
            start_x = face_center_x - dynamic_side // 2
            start_y = face_center_y - int(dynamic_side * 0.33)
            
        else:
            dynamic_side = min(h_orig, w_orig)
            start_x = w_orig // 2 - dynamic_side // 2
            start_y = h_orig // 2 - dynamic_side // 2
            fallback_count += 1
            
        start_x = max(0, min(start_x, w_orig - dynamic_side))
        start_y = max(0, min(start_y, h_orig - dynamic_side))
        
        square_img = img[start_y:start_y+dynamic_side, start_x:start_x+dynamic_side]
        final_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_aligned.png")
        
        cv2.imwrite(output_path, final_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        
        # 记录真实的面部占比，用于后续的平均值计算
        actual_ratio = round(w / dynamic_side, 3) if faces else 0
        
        metadata_list.append({
            "id": name,
            "group": group_name,
            "orig_w": w_orig,   
            "orig_h": h_orig,   
            "face_ratio_w": actual_ratio, 
            "y_true": 0,        
            "file_path": output_path
        })
        
        success_count += 1

    print(f"   ✅ 成功入库: {success_count} | ❌ 防上采样丢弃: {discard_need_upsample} | 🚑 大头满充抢救: {salvage_too_large} | ⚠️ 盲切: {fallback_count}")
    return success_count

if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    base_data_dir = os.path.join(project_root, "data", "real_samples")
    metadata_out_path = os.path.join(project_root, "data", "metadata_real_raw.csv")

    groups = ["male-doctor", "female-doctor", "male-nurse", "female-nurse"]
    all_metadata = []

    print("="*60)
    print("🚀 开始执行【非动态定比 + 最大满充抢救】清洗管线...")
    print("="*60)

    for group in groups:
        input_folder = os.path.join(base_data_dir, group)
        output_folder = os.path.join(base_data_dir, f"{group}_after")
        
        if os.path.exists(input_folder):
            process_real_images_strict(input_folder, output_folder, all_metadata, group)
        else:
            print(f"   [跳过] 找不到目录: {input_folder}")

    if all_metadata:
        with open(metadata_out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f"\n🎉 真实数据清洗完毕！数据已写入 CSV。")