# ==========================================
# BƯỚC 4: DUỖI THẲNG NHIỄM SẮC THỂ
# CẢI TIẾN: Tự động tạo mask nếu chưa có + Skeleton-based alignment
# ==========================================

import cv2
import numpy as np
from pathlib import Path
from medical_pipeline.morphology.alignment import align_chromosome


def _create_mask_from_bgr(img_bgr: np.ndarray) -> np.ndarray:
    """
    Tạo mask nhị phân từ ảnh BGR.
    Logic: Pixel tối (NST) = True, Pixel sáng (nền trắng) = False.
    
    Cải tiến từ Notebook (Cell 58-63): create_filled_mask
    """
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr
    
    # Adaptive threshold cho trường hợp ảnh không đều sáng
    # Dùng Otsu để tự động chọn ngưỡng tối ưu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Morphology: đóng lỗ nhỏ bên trong NST
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Loại nhiễu nhỏ
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    return binary > 0


def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
    """Đảm bảo ảnh là grayscale uint8."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def run_straightening(chromosomes: list, report_dir: Path = None):
    """
    Duỗi thẳng tất cả NST trong danh sách.
    
    Cải tiến:
    - Tự tạo mask nếu chưa có (từ ảnh BGR bằng Otsu threshold)
    - Dùng ARAP Puppet Warp (Skeleton-based) để uốn nắn NST cong (Step 4 Advanced)
    - Chuyển ảnh sang grayscale nếu đang là BGR
    - Lưu ảnh duỗi thẳng vào report_dir
    """
    print("Chay logic xoay thang (alignment) NST...")
    
    from medical_pipeline.morphology.arap_straightening import apply_arap_straightening
    
    out_path = None
    if report_dir:
        out_path = report_dir / "step4_straightened"
        out_path.mkdir(parents=True, exist_ok=True)
        
    for chrom in chromosomes:
        img = chrom.get("image", None)
        if img is None:
            continue
            
        # === TỰ ĐỘNG TẠO MASK NẾU CHƯA CÓ ===
        mask = chrom.get("mask", None)
        if mask is None:
            mask = _create_mask_from_bgr(img)
            chrom["mask"] = mask
            
        # Đảm bảo image là grayscale cho alignment
        gray_img = _ensure_grayscale(img)
        
        # Đảm bảo mask là boolean
        if mask.dtype != bool:
            mask = mask > 127 if mask.max() > 1 else mask > 0
            
        try:
            # 1. Bẻ thẳng NST cong bằng ARAP (Puppet Warp)
            gray_img, mask = apply_arap_straightening(gray_img, mask)
            
            # 2. Gọi pipeline căn chỉnh hoàn chỉnh (Xoay dọc + Tìm Centromere)
            aligned_img, aligned_mask, metadata = align_chromosome(gray_img, mask)
            
            # Cập nhật vào dictionary
            chrom["straight_image"] = aligned_img
            chrom["aligned_image"] = aligned_img
            chrom["aligned_mask"] = aligned_mask
            chrom["alignment_metadata"] = metadata
            
            if out_path:
                # Lưu ảnh duỗi thẳng
                stem = chrom.get('stem', f'chrom_{id(chrom)}')
                cv2.imwrite(str(out_path / f"{stem}_aligned.png"), aligned_img)
                
        except Exception as e:
            print(f"  ⚠️ Lỗi duỗi thẳng {chrom.get('stem', '?')}: {e}")
            # Fallback: dùng ảnh gốc
            chrom["straight_image"] = gray_img
            chrom["aligned_image"] = gray_img
            chrom["aligned_mask"] = mask
            
    return chromosomes
