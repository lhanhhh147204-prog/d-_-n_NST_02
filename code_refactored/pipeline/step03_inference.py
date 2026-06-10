# ============================================================
# FILE: pipeline/03_inference.py
# CHỨC NĂNG: Dự đoán ảnh thực tế và inpaint tách A/B
# ============================================================

import os
from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

from config.settings import (
    RAW_OVERLAP_DIR, RESULTS_DIR, IMG_SIZE, ABC_THRESHOLDS
)
from core.image_utils import (
    resize_with_padding, restore_mask_to_original, 
    clean_mask, make_overlay, rgba_from_instance
)
from core.model_unet import hybrid_loss, student_distill_loss

def prepare_input(img: Image.Image) -> np.ndarray:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)

def save_binary_mask(mask: np.ndarray, path: Path):
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(path)

def save_separated_ab(original_gray: Image.Image, stem: str, mask_A: np.ndarray, mask_B: np.ndarray, mask_C: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    gray = np.array(original_gray.convert("L"))

    A_full = clean_mask(mask_A | mask_C, keep_components=1)
    B_full = clean_mask(mask_B | mask_C, keep_components=1)
    C_clean = clean_mask(mask_C, keep_components=1)

    img_A = rgba_from_instance(gray, A_full, C_clean, transparent=True)
    img_B = rgba_from_instance(gray, B_full, C_clean, transparent=True)

    img_A.save(out_dir / f"{stem}_A.png")
    img_B.save(out_dir / f"{stem}_B.png")

def run_inference(input_dir=RAW_OVERLAP_DIR, model_path=None):
    if model_path is None:
        model_path = RESULTS_DIR / "best_student.keras"
        if not model_path.exists():
            model_path = RESULTS_DIR / "best_teacher.keras"
            
    if not model_path.exists():
        print(f"❌ Không tìm thấy model tại {model_path}")
        return

    out_dir = RESULTS_DIR / "real_predictions" / "separated_chromosomes"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {model_path}")
    # Load với custom_objects để không lỗi khi parse loss function dù compile=False
    custom_objects = {
        "hybrid_loss": hybrid_loss, 
        "student_distill_loss": student_distill_loss
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    images = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg"))
    if not images:
        print(f"⚠️ Không có ảnh nào trong {input_dir}")
        return

    print(f"Bắt đầu dự đoán {len(images)} ảnh...")
    for img_path in images:
        stem = img_path.stem
        original_img = Image.open(img_path).convert("L")
        
        processed_img, meta = resize_with_padding(original_img, IMG_SIZE, is_mask=False)
        x = prepare_input(processed_img)
        pred = model.predict(x, verbose=0)[0]

        mask_A_256 = clean_mask(pred[..., 0] > ABC_THRESHOLDS[0])
        mask_B_256 = clean_mask(pred[..., 1] > ABC_THRESHOLDS[1])
        mask_C_256 = clean_mask(pred[..., 2] > ABC_THRESHOLDS[2])

        mask_A = restore_mask_to_original(mask_A_256, meta)
        mask_B = restore_mask_to_original(mask_B_256, meta)
        mask_C = restore_mask_to_original(mask_C_256, meta)

        save_separated_ab(original_img, stem, mask_A, mask_B, mask_C, out_dir)
        
    print(f"✅ Hoàn thành dự đoán. Kết quả lưu tại: {out_dir}")

if __name__ == "__main__":
    run_inference()
