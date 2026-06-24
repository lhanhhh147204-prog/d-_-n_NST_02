# ==========================================
# BƯỚC 3: MODULE ĐIỀU PHỐI (ROUTER)
# CHỨC NĂNG: Đưa cụm ảnh vào U-Net để phân loại và cắt
# ==========================================

import cv2
import numpy as np
from pathlib import Path
import warnings

from medical_pipeline.pipeline.buoc3_0_phan_loai_cum import ClusterClassifier
from medical_pipeline.pipeline.buoc3_1_xu_ly_cham import process_touching_cluster
from medical_pipeline.pipeline.buoc3_2_xu_ly_chong import process_overlapping_cluster

def run_overlap_separation(input_dir: Path, output_dir: Path = None) -> list:
    """
    Chạy toàn bộ pipeline Bước 3: Phân loại & Tách cụm theo kiến trúc Notebook gốc.
    """
    print(f"\n[BƯỚC 3] Phân loại & Xử lý cụm NST...")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    final_separated_chromosomes = []
    
    valid_exts = {'.png', '.jpg', '.jpeg'}
    image_files = [p for p in input_dir.iterdir() if p.suffix.lower() in valid_exts]
    
    if not image_files:
        print(f"⚠️ Không tìm thấy ảnh cụm nào trong {input_dir}")
        return []
        
    print(f"Tiến hành phân loại và tách {len(image_files)} cụm NST...")
    
    # Khởi tạo mô hình phân loại cụm (DualBranchModel)
    weights_path = Path("ai_training/weights/improved_model_swin_rn50fpnv2.pth")
    if not weights_path.exists():
        alt_path = Path("ai_training/weights/best_model_epoch.pth")
        if alt_path.exists():
            weights_path = alt_path
            
    classifier = ClusterClassifier(weights_path)
    
    from collections import deque
    queue = deque()
    
    for img_path in image_files:
        img_array = np.fromfile(str(img_path), np.uint8)
        img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img_bgr is not None:
            queue.append({
                "img": img_bgr,
                "stem": img_path.stem,
                "depth": 0
            })
            
    while queue:
        item = queue.popleft()
        img_bgr = item["img"]
        stem = item["stem"]
        depth = item["depth"]
            
        # 1. PHÂN LOẠI
        cluster_class = classifier.predict(img_bgr)
        
        separated_parts = []
        
        # 2. ĐỊNH TUYẾN (ROUTER)
        if cluster_class == 0 or depth >= 3:
            # Lớp 0 (Đơn) hoặc quá giới hạn đệ quy: Giữ nguyên
            if cluster_class == 0:
                print(f"  -> {stem}.png: Lớp Đơn (0). Bỏ qua cắt.")
            else:
                print(f"  -> {stem}.png: Đạt giới hạn đệ quy ({depth}). Bỏ qua.")
            separated_parts = [img_bgr]
            
        elif cluster_class == 1:
            print(f"  -> {stem}.png: Lớp Chạm (1). Cắt hình học (Depth {depth}).")
            parts = process_touching_cluster(img_bgr)
            separated_parts = parts if parts else [img_bgr]
            
        elif cluster_class == 2:
            print(f"  -> {stem}.png: Lớp Chồng (2). Tách bằng U-Net (Depth {depth}).")
            parts = process_overlapping_cluster(img_bgr)
            separated_parts = parts if parts else [img_bgr]
            
        elif cluster_class == 3:
            print(f"  -> {stem}.png: Lớp Dị vật (3). Bỏ qua.")
            continue
            
        # 3. QUYẾT ĐỊNH ĐỆ QUY HAY LƯU
        # Tính toán diện tích thực của các mảnh (bỏ qua nền trắng 255)
        def is_valid_size(part_img):
            if len(part_img.shape) == 3:
                gray = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)
            else:
                gray = part_img
            # Đếm số pixel không phải trắng (NST thường tối màu hơn nền trắng)
            return np.sum(gray < 250) > 100

        # 3. LƯU KẾT QUẢ (KHÔNG ĐỆ QUY)
        # Giống y hệt Notebook gốc: sau khi cắt 1 lần, các mảnh được coi là NST đơn và lưu luôn.
        # Không đưa lại vào queue để phân loại vì model có thể nhận diện nhầm ảnh đơn thành cụm chạm rồi cắt nát.
        for idx, part in enumerate(separated_parts):
            if is_valid_size(part):
                # Lưu dưới dạng NST đơn
                stem_name = f"{stem}_part_{idx}" if len(separated_parts) > 1 else stem
                chrom_data = {
                    "stem": stem_name,
                    "image": part,
                    "original_cluster_class": cluster_class 
                }
                final_separated_chromosomes.append(chrom_data)
                
                if output_dir:
                    is_success, im_buf_arr = cv2.imencode('.png', part)
                    if is_success:
                        im_buf_arr.tofile(str(output_dir / f"{stem_name}.png"))
            else:
                part_name = f"{stem}_part_{idx}" if len(separated_parts) > 1 else stem
                print(f"  -> {part_name}.png: Mảnh quá nhỏ (rác). Bỏ qua.")
                
    print(f"[BƯỚC 3 HOÀN TẤT] Trích xuất được {len(final_separated_chromosomes)} NST đơn.")
    return final_separated_chromosomes
