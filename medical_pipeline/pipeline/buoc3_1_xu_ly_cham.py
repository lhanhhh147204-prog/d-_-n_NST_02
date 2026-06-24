# ==========================================
# BƯỚC 3.1: XỬ LÝ CỤM CHẠM (TOUCHING)
# CHỨC NĂNG: Tách các cụmNST chỉ chạm nhau ở viền
# ==========================================

import cv2
import numpy as np
from medical_pipeline.morphology.geometric_cutter import (
    find_concave_points,
    find_best_cutting_points,
    process_chromosome_image,
    compute_separation_path,
    split_and_return_images
)

def process_touching_cluster(image_bgr: np.ndarray) -> list:
    """
    Xử lý NST chạm (Class 1) sử dụng thuật toán vi điểm lõm hình học (Notebook gốc).
    """
    print("[buoc3_1] Đang xử lý cụm CHẠM (Touch) bằng Điểm Lõm Hình Học...")
    
    # 1. Tìm điểm lõm
    concavity_points, _, _, original_polygon = find_concave_points(image_bgr)
    if not concavity_points or not original_polygon:
        return [image_bgr] # Fallback
        
    # 2. Tìm cặp điểm cắt tốt nhất
    ideal_pairs = find_best_cutting_points(concavity_points, image_bgr)
    if not ideal_pairs:
        return [image_bgr]

    # 3. Xử lý ảnh để tạo opening mask
    img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    opening_result = process_chromosome_image(img_gray)
    if opening_result is None:
        return [image_bgr]
    
    if img_gray.shape != opening_result.shape:
        opening_result = cv2.resize(opening_result, (img_gray.shape[1], img_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 4. Tìm đường cắt tối ưu tạo ra 2 mảnh HỢP LỆ (không phải rác)
    from medical_pipeline.morphology.geometric_cutter import draw_and_count_white_pixels
    
    path_candidates = []
    for point1, point2 in ideal_pairs:
        separation_path = compute_separation_path(point1, point2, opening_result, img_gray)
        if separation_path:
            white_count, _ = draw_and_count_white_pixels(separation_path, opening_result)
            path_candidates.append({
                "path": separation_path,
                "white_count": white_count
            })
            
    # Sắp xếp các đường cắt ưu tiên đường đi qua ít vùng trắng nhất
    path_candidates.sort(key=lambda x: x["white_count"])
    
    # Hàm kiểm tra mảnh có hợp lệ không (giống buoc3_main)
    def is_valid_size(part_img):
        part_gray = cv2.cvtColor(part_img, cv2.COLOR_BGR2GRAY)
        return np.sum(part_gray < 250) > 100

    # 5. Thử từng đường cắt, nếu tạo ra 2 mảnh đều LỚN (không phải rác cắt góc) thì lấy!
    for candidate in path_candidates:
        part1, part2 = split_and_return_images(candidate["path"], original_polygon, image_bgr)
        if part1 is not None and part2 is not None:
            if is_valid_size(part1) and is_valid_size(part2):
                return [part1, part2]
                
    return [image_bgr]

