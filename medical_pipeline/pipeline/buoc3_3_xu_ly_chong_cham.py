# ==========================================
# BƯỚC 3.3: XỬ LÝ CỤM CHỒNG CHẠM (COMPLEX)
# CHỨC NĂNG: Xử lý các cụm phức hợp
# ==========================================

import cv2
import numpy as np
from medical_pipeline.pipeline.buoc3_1_xu_ly_cham import process_touching_cluster
from medical_pipeline.pipeline.buoc3_2_xu_ly_chong import process_overlapping_cluster

def process_complex_cluster(image_bgr: np.ndarray, mask: np.ndarray = None) -> list:
    """
    Xử lý NST phức hợp vừa chồng vừa chạm (Class 3).
    Quy trình:
    1. Tìm điểm lõm cắt geometric (giống Class 2) để phá vỡ khối lớn.
    2. Các khối nhỏ tiếp tục quét xem còn dính viền không (dùng Class 1).
    
    Input: Ảnh BGR của cụm.
    Output: Danh sách các ảnh con.
    """
    print("[buoc3_3] Đang xử lý cụm PHỨC TẠP (Chồng Chạm)...")
    
    # Bước 1: Giảm thiểu độ chồng
    partially_separated = process_overlapping_cluster(image_bgr, mask)
    
    # Bước 2: Tách nốt phần chạm
    final_results = []
    for part in partially_separated:
        final_results.extend(process_touching_cluster(part))
        
    return final_results
