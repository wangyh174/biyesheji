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

print("正在加载 MTCNN 模型 (当前设定：导师绿灯版 - 动态弹性搜索机制)...")
detector = MTCNN()

def process_real_images_ultimate(input_dir, output_dir, metadata_list, group_name, target_size=512):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = glob(os.path.join(input_dir, '*.*'))
    success_count = 0
    fallback_count = 0 
    error_count = 0 
    
    discard_orig_small = 0   
    discard_need_upsample = 0 
    discard_too_large = 0     

    print(f"--> 正在处理组别: {group_name} | 候选总数: {len(image_paths)}")

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None: 
            error_count += 1
            continue
        
        h_orig, w_orig = img.shape[:2]

        # 如果原图连 512x512 都没有，直接抛弃（物理底线）
        if h_orig < target_size or w_orig < target_size:
            discard_orig_small += 1
            continue

        # 防 OOM 的缩图检测机制 (只缩放用于 MTCNN 找脸，不影响最终截取)
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
            # 选取画面中最大的脸
            face = max(faces, key=lambda rect: rect['box'][2] * rect['box'][3])
            
            # 将人脸坐标还原到原图真实尺寸
            x, y, w, h = [int(v / scale_factor) for v in face['box']]
            face_center_x = x + w // 2
            face_center_y = y + h // 2
            
            # =========================================================
            # 🚀 导师绿灯版：动态弹性比例搜索 (兼顾远景全景与特写大头)
            # =========================================================
            # 定义容忍比例阶梯：从最完美的 0.35 开始尝试，逐渐向特写(0.55)和远景(0.2)妥协
            allowed_ratios = [0.35, 0.33, 0.37, 0.3, 0.4, 0.28, 0.42, 0.25, 0.45, 0.2, 0.5, 0.55]
            
            
            dynamic_side = None
            final_ratio = None
            
            for ratio in allowed_ratios:
                test_side = int(w / ratio)
                
                # 检查条件 1：不能引发放大插值（保护高频物理底噪）
                if test_side < target_size:
                    continue 
                # 检查条件 2：不能超出原图物理边界（保护真实像素，不补黑边）
                if test_side > min(h_orig, w_orig):
                    continue 
                    
                # 两个条件都满足，说明这个比例完美适用当前这张图！
                dynamic_side = test_side
                final_ratio = ratio
                break # 找到合适的比例，立刻停止搜索
                
            # 如果把阶梯全试了一遍都不行，说明极端到无可救药，只能丢弃
            if dynamic_side is None:
                if int(w / 0.35) < target_size:
                    discard_need_upsample += 1 # 脸实在太小
                else:
                    discard_too_large += 1     # 脸实在太大，连 0.55 都装不下
                continue
                
            # =========================================================
            # 动态调整 Y 轴起刀点 (防止大头照被切掉下巴)
            # =========================================================
            # 优雅的线性公式：脸越小起刀点越靠上(留出胸口)，脸越大起刀点越居中(保护整脸)
            y_offset_ratio = 0.5 - (final_ratio * 0.5) 
            
            start_x = face_center_x - dynamic_side // 2
            start_y = face_center_y - int(dynamic_side * y_offset_ratio)
            
        else:
            # 没找到脸的盲切兜底（从正中间切最大的正方形）
            dynamic_side = min(h_orig, w_orig)
            start_x = w_orig // 2 - dynamic_side // 2
            start_y = h_orig // 2 - dynamic_side // 2
            fallback_count += 1
            final_ratio = 0 # 记录为 0 代表是盲切的
            
        # 边界截断保护 (最后一道防线，确保裁切框绝对不越界)
        start_x = max(0, min(start_x, w_orig - dynamic_side))
        start_y = max(0, min(start_y, h_orig - dynamic_side))
        
        # 纯物理平移下刀，取出完美正方形
        square_img = img[start_y:start_y+dynamic_side, start_x:start_x+dynamic_side]

        # 降采样回标准的 512x512 (使用最安全的 INTER_AREA 算法)
        final_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)

        # 保存为绝对无损的 PNG 格式
        filename = os.path.basename(img_path)
        name, _ = os.path.splitext(filename)
        output_path = os.path.join(output_dir, f"{name}_aligned.png")
        cv2.imwrite(output_path, final_img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
        
        # =========================================================
        # 📊 终极审计防线：记录最终采用的面部占比，用于计算组间公平性
        # =========================================================
        metadata_list.append({
            "id": name,
            "group": group_name,
            "orig_w": w_orig,   
            "orig_h": h_orig,   
            "face_ratio_w": final_ratio, # 这个值极其重要，决定了你的答辩生死
            "y_true": 0,        
            "file_path": output_path
        })
        
        success_count += 1

    print(f"   ✅ 成功入库: {success_count} | ❌ 防上采样丢弃: {discard_need_upsample} | ❌ 极端大头越界丢弃: {discard_too_large} | ⚠️ 盲切: {fallback_count}")
    return success_count

if __name__ == "__main__":
    # 自动获取脚本当前所在目录
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    # 这里假设你的图片放在上一级目录的 data/real_samples 下
    base_data_dir = os.path.join(project_root, "data", "real_samples")
    metadata_out_path = os.path.join(project_root, "data", "metadata_real_raw.csv")

    # 你需要处理的四个组别文件夹名称
    groups = ["male-doctor", "female-doctor", "male-nurse", "female-nurse"]
    all_metadata = []

    print("="*70)
    print("🚀 开始执行【导师绿灯版：动态弹性对齐】清洗管线...")
    print("="*70)

    for group in groups:
        input_folder = os.path.join(base_data_dir, group)
        output_folder = os.path.join(base_data_dir, f"{group}_after")
        
        if os.path.exists(input_folder):
            process_real_images_ultimate(input_folder, output_folder, all_metadata, group)
        else:
            print(f"   [跳过] 找不到目录: {input_folder}")

    if all_metadata:
        # 将结果写入 CSV，方便你做统计学平均值核对
        with open(metadata_out_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_metadata[0].keys())
            writer.writeheader()
            writer.writerows(all_metadata)
        print(f"\n🎉 真实数据基准构建完毕！审计日志已安全写入: \n   {metadata_out_path}")
        print("\n💡 下一步行动指南：")
        print("请打开 CSV 文件，分别计算四个组别的 `face_ratio_w` 的平均值。")
        print("只要这四个平均值相近，你就可以理直气壮地对导师说：我已经控制好了属性间的构图变量！")
    else:
        print("\n⚠️ 未产生任何有效数据，请检查输入目录及路径设置。")