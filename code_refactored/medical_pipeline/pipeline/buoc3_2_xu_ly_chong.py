# ==========================================
# BƯỚC 3.2: XỬ LÝ CỤM CHỒNG (OVERLAPPING) - KIẾN TRÚC MỚI U-NET
# CHỨC NĂNG: Sử dụng mô hình U-Net (Teacher) để nhận diện vùng chồng và cắt NST
# CẢI TIẾN LẦN 3: Tích hợp đệm nền ẩn (Hidden Padding) để chống vỡ hình.
# ==========================================

import cv2
import numpy as np
from pathlib import Path
import warnings
import tensorflow as tf
from tensorflow import keras

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from config.settings import IMG_SIZE, WEIGHTS_DIR, ABC_THRESHOLDS
from medical_pipeline.core.models.losses import hybrid_loss, student_distill_loss

def crop_to_content(img_bgr: np.ndarray, mask_gray: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    
    pad = 5
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(img_bgr.shape[1] - x, w + 2 * pad)
    h = min(img_bgr.shape[0] - y, h + 2 * pad)
    
    return img_bgr[y:y+h, x:x+w]

def pad_to_square(img_bgr: np.ndarray):
    """
    Đệm nền trắng tạo thành hình vuông cơ động để không làm méo tỷ lệ sinh học khi đưa vào U-Net.
    Trả về: ảnh padded, padding offsets (y_off, x_off), size gốc (h, w)
    """
    h, w = img_bgr.shape[:2]
    side_len = max(w, h)
    
    # Nền trắng
    padded_img = np.ones((side_len, side_len, 3), dtype=np.uint8) * 255
    y_off = (side_len - h) // 2
    x_off = (side_len - w) // 2
    
    padded_img[y_off:y_off+h, x_off:x_off+w] = img_bgr
    return padded_img, y_off, x_off, h, w

class UNetSeparator:
    def __init__(self, model_path: Path):
        self.has_weights = model_path.exists()
        if not self.has_weights:
            warnings.warn(f"⚠️ [CẢNH BÁO] Không tìm thấy U-Net tại {model_path}. Sẽ bỏ qua chia tách.")
            self.model = None
            return
            
        print(f"Loading mô hình Keras U-Net (Cắt Chồng Chéo) từ {model_path}...")
        custom_objects = {
            "hybrid_loss": hybrid_loss,
            "student_distill_loss": student_distill_loss,
        }
        self.model = keras.models.load_model(str(model_path), custom_objects=custom_objects, compile=False)

    def separate(self, img_bgr: np.ndarray) -> list:
        if self.model is None:
            return [img_bgr]
            
        # 1. Đệm nền (Padding) thay vì resize trực tiếp để tránh vỡ ảnh
        padded_img, y_off, x_off, orig_h, orig_w = pad_to_square(img_bgr)
        
        # 2. Tiền xử lý trên ảnh Padded
        gray = cv2.cvtColor(padded_img, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        
        img_tensor = tf.convert_to_tensor(gray_resized, dtype=tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=-1)
        img_tensor = img_tensor / 255.0
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        # 3. Dự đoán
        preds = self.model.predict(img_tensor, verbose=0)[0]
        
        pred_a = preds[..., 0] > ABC_THRESHOLDS[0]
        pred_b = preds[..., 1] > ABC_THRESHOLDS[1]
        
        if np.sum(pred_a) < 50 or np.sum(pred_b) < 50:
            return [img_bgr]
            
        # 4. Phóng to Mask về đúng kích thước Padded vuông
        padded_h, padded_w = padded_img.shape[:2]
        mask_a_uint8 = (pred_a * 255).astype(np.uint8)
        mask_b_uint8 = (pred_b * 255).astype(np.uint8)
        
        mask_a_padded = cv2.resize(mask_a_uint8, (padded_w, padded_h), interpolation=cv2.INTER_NEAREST)
        mask_b_padded = cv2.resize(mask_b_uint8, (padded_w, padded_h), interpolation=cv2.INTER_NEAREST)
        
        # 5. Gỡ nền trắng (Un-Pad) để trả lại kích thước sát rạt gốc
        mask_a_orig = mask_a_padded[y_off:y_off+orig_h, x_off:x_off+orig_w]
        mask_b_orig = mask_b_padded[y_off:y_off+orig_h, x_off:x_off+orig_w]
        
        # 6. Tách mảng NST bằng Inpainting và Alpha Blending (Distance Transform)
        dist_a = cv2.distanceTransform(mask_a_orig, cv2.DIST_L2, 5)
        dist_b = cv2.distanceTransform(mask_b_orig, cv2.DIST_L2, 5)
        
        cv2.normalize(dist_a, dist_a, 0, 1.0, cv2.NORM_MINMAX)
        cv2.normalize(dist_b, dist_b, 0, 1.0, cv2.NORM_MINMAX)
        
        alpha_a = cv2.merge([dist_a, dist_a, dist_a])
        alpha_b = cv2.merge([dist_b, dist_b, dist_b])
        
        # Inpaint (TELEA) để khôi phục vùng bị che
        inpaint_mask = cv2.bitwise_and(mask_a_orig, mask_b_orig)
        inpainted_img = cv2.inpaint(img_bgr, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        
        white_bg = np.ones_like(img_bgr, dtype=np.float32) * 255
        inpainted_float = inpainted_img.astype(np.float32)
        
        part_a = (inpainted_float * alpha_a + white_bg * (1 - alpha_a)).astype(np.uint8)
        part_b = (inpainted_float * alpha_b + white_bg * (1 - alpha_b)).astype(np.uint8)
        
        # Cắt gọt rìa thừa lần cuối (vẫn cần threshold nhẹ để xác định biên)
        mask_a_final = (dist_a > 0.05).astype(np.uint8) * 255
        mask_b_final = (dist_b > 0.05).astype(np.uint8) * 255
        
        part_a_cropped = crop_to_content(part_a, mask_a_final)
        part_b_cropped = crop_to_content(part_b, mask_b_final)
        
        return [part_a_cropped, part_b_cropped]

_separator_instance = None

def get_unet_separator():
    global _separator_instance
    if _separator_instance is None:
        model_path = WEIGHTS_DIR / "best_for_apply_inference.keras"
        _separator_instance = UNetSeparator(model_path)
    return _separator_instance

def process_overlapping_cluster(image_bgr: np.ndarray) -> list:
    separator = get_unet_separator()
    return separator.separate(image_bgr)
