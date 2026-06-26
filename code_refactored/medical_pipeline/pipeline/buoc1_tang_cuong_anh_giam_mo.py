# ==========================================
# BƯỚC 1: TĂNG CƯỜNG ẢNH Y KHOA (Clinical Enhancement)
# CHỨC NĂNG: Khử mờ, tăng tương phản, giữ lại vằn G-bands
# ==========================================

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def apply_clinical_enhancement(img_bgr: np.ndarray) -> np.ndarray:
    """
    Áp dụng bộ lọc truyền thống chuẩn y khoa thay cho MPRNet:
    1. Bilateral Filter: Khử nhiễu nền (noise) nhưng GIỮ NGUYÊN độ sắc nét của cạnh.
    2. CLAHE: Tăng tương phản cục bộ để làm nổi bật dải G-bands.
    3. Unsharp Mask: Tăng cường độ bén ở vùng viền.
    """
    # 1. Khử nhiễu hạt bằng Bilateral Filter
    # d=9, sigmaColor=75, sigmaSpace=75 là thông số chuẩn để giữ viền
    smoothed = cv2.bilateralFilter(img_bgr, 9, 75, 75)

    # 2. CLAHE trên kênh L (Lightness) của không gian LAB
    lab = cv2.cvtColor(smoothed, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 3. Unsharp Masking để sắc nét viền
    gaussian = cv2.GaussianBlur(enhanced_bgr, (5, 5), 2.0)
    # Công thức: sharpened = original + 1.5 * (original - blurred)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.5, gaussian, -0.5, 0)
    
    return sharpened

def run_deblur(original_img_bgr: np.ndarray) -> np.ndarray:
    """
    Hàm đầu mối được gọi từ buoc9_tong_hop.py.
    """
    print("  [BƯỚC 1] Đang áp dụng chuỗi lọc Clinical Enhancement (Bilateral + CLAHE + Unsharp Mask)...")
    enhanced = apply_clinical_enhancement(original_img_bgr)
    return enhanced

def compare_color_structure(original_img, deblurred_img):
    import matplotlib.pyplot as plt
    if original_img.shape != deblurred_img.shape:
        deblurred_img = cv2.resize(deblurred_img, (original_img.shape[1], original_img.shape[0]))

    orig_gray = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    deb_gray = cv2.cvtColor(deblurred_img, cv2.COLOR_BGR2GRAY)
    ssim_value, ssim_map = ssim(orig_gray, deb_gray, full=True)

    orig_lab = cv2.cvtColor(original_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    deb_lab = cv2.cvtColor(deblurred_img, cv2.COLOR_BGR2LAB).astype(np.float32)
    delta_e = np.linalg.norm(orig_lab - deb_lab, axis=2)
    average_delta_e = np.mean(delta_e)

    print(f"Giá trị SSIM: {ssim_value:.4f}")
    print(f"Giá trị Delta E trung bình: {average_delta_e:.2f}")
    return ssim_value, average_delta_e
