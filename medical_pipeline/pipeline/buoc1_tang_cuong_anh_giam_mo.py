# ==========================================
# CODE TRÍCH XUẤT TỪ NOTEBOOK - BƯỚC 1
# CHỨC NĂNG: Khử mờ ảnh
# ==========================================

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

# Kéo MPRNet từ source code đã có trong project
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "core" / "models" / "MPRNet"))
try:
    from MPRNet import MPRNet
except ImportError:
    pass # Sẽ xử lý sau nếu thiếu thư viện

class NotebookDeblurrer:
    def __init__(self, weights_path: str, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = MPRNet().to(self.device)
            self._load_weights(weights_path)
            self.model = nn.DataParallel(self.model)
            self.model.eval()
        except:
            self.model = None

    def _load_weights(self, weights_path: str):
        ckpt = torch.load(weights_path, map_location=self.device)
        state_dict = ckpt.get('state_dict', ckpt)
        cleaned = {k.replace('module.', ''):v for k,v in state_dict.items()}
        self.model.load_state_dict(cleaned)

    def deblur(self, image_input, return_bgr: bool = True):
        if self.model is None:
            return image_input # Fallback nếu không có model
            
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        else:
            img = image_input.copy()
        
        # Đảm bảo ảnh là BGR (ảnh màu) vì model cần 3 kênh
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        h,w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        tensor = torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).to(self.device)

        pad_h = ((h+7)//8)*8 - h
        pad_w = ((w+7)//8)*8 - w
        if pad_h>0 or pad_w>0:
            tensor = F.pad(tensor, (0,pad_w,0,pad_h), mode='reflect')

        with torch.no_grad():
            out = torch.clamp(self.model(tensor)[0],0,1).squeeze(0)

        if pad_h>0 or pad_w>0:
            out = out[:, :h, :w]

        out_np = out.permute(1,2,0).cpu().numpy()
        out_uint8 = img_as_ubyte(out_np)
        return cv2.cvtColor(out_uint8, cv2.COLOR_RGB2BGR) if return_bgr else out_uint8

def apply_clinical_enhancement(img_bgr: np.ndarray) -> np.ndarray:
    """
    Áp dụng CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    và Unsharp Mask để làm nổi bật dải băng G-bands trên NST.
    """
    # 1. CLAHE trên kênh L (Lightness) của không gian LAB
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_bgr = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 2. Unsharp Masking để sắc nét viền
    gaussian = cv2.GaussianBlur(enhanced_bgr, (5, 5), 2.0)
    # Công thức Unsharp Mask: img = img + amount * (img - blurred)
    # Viết lại dưới dạng cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.5, gaussian, -0.5, 0)
    
    return sharpened

def run_deblur(original_img_bgr):
    weights = str(project_root / "weights" / "deblurring_best.pth")
    deb = NotebookDeblurrer(weights)
    
    if deb.model is None:
        print("  ⚠️ Không tìm thấy MPRNet model. Áp dụng chuẩn tăng cường y khoa (CLAHE + Unsharp Mask)...")
        return apply_clinical_enhancement(original_img_bgr)
        
    return deb.deblur(original_img_bgr)

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
