# ============================================================
# FILE: data/preprocessing.py
# CHỨC NĂNG: Logic xử lý ảnh NST thô -> tạo dữ liệu tổng hợp
# TÁCH TỪ: pipeline/step01_data_prep.py (phần hàm nghiệp vụ)
# ============================================================

import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

from config.settings import (
    BG_MARGIN, CROP_PADDING, KERNEL_SIZE, MASK_SHRINK_ITER,
    CANVAS_SIZE, TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX,
    MIN_OVERLAP_PIXELS, MAX_OVERLAP_RATIO,
)
from morphology.mask_ops import get_border_background, keep_largest_components


def extract_object_rgba(img_path: Path) -> tuple:
    """Tách NST đơn ra khỏi nền trắng, trả về (ảnh RGBA, mask) hoặc (None, None) nếu thất bại."""
    img = Image.open(img_path).convert("RGB")
    gray = np.array(img.convert("L"))

    background, threshold, bg_ref = get_border_background(gray, BG_MARGIN)
    object_mask = ~background

    kernel = np.ones((KERNEL_SIZE, KERNEL_SIZE), np.uint8)
    object_mask = cv2.morphologyEx(object_mask.astype(np.uint8)*255, cv2.MORPH_OPEN, kernel)
    object_mask = cv2.morphologyEx(object_mask, cv2.MORPH_CLOSE, kernel) > 0
    object_mask = keep_largest_components(object_mask, keep=1)

    if MASK_SHRINK_ITER > 0:
        object_mask = cv2.erode(object_mask.astype(np.uint8)*255, kernel, iterations=MASK_SHRINK_ITER) > 0

    ys, xs = np.where(object_mask)
    if len(ys) == 0:
        return None, None
    x1, x2 = max(0, xs.min()-CROP_PADDING), min(img.width, xs.max()+CROP_PADDING)
    y1, y2 = max(0, ys.min()-CROP_PADDING), min(img.height, ys.max()+CROP_PADDING)

    cropped_mask = object_mask[y1:y2, x1:x2]
    cropped_img = np.array(img)[y1:y2, x1:x2]

    rgba = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = cropped_img
    rgba[..., 3] = cropped_mask.astype(np.uint8) * 255

    return Image.fromarray(rgba, mode="RGBA"), Image.fromarray(cropped_mask.astype(np.uint8)*255, mode="L")


def _rotate_pair(img: Image.Image, mask: Image.Image, angle: float) -> tuple:
    """Xoay đồng bộ cặp (ảnh, mask) một góc nhất định."""
    return (
        img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255,0)),
        mask.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)
    )


def make_overlap_canvas(obj_A, mask_A, obj_B, mask_B):
    """
    Ghép 2 NST tạo vùng overlap (C = A ∩ B).
    Trả về (ảnh_RGB, mask_A, mask_B, mask_C) hoặc None nếu overlap không hợp lệ.
    """
    tA = random.randint(TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX)
    tB = random.randint(TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX)

    scale_A = min(tA/obj_A.width, tA/obj_A.height)
    scale_B = min(tB/obj_B.width, tB/obj_B.height)

    obj_A = obj_A.resize((int(obj_A.width*scale_A), int(obj_A.height*scale_A)), Image.BILINEAR)
    mask_A = mask_A.resize(obj_A.size, Image.NEAREST)
    obj_B = obj_B.resize((int(obj_B.width*scale_B), int(obj_B.height*scale_B)), Image.BILINEAR)
    mask_B = mask_B.resize(obj_B.size, Image.NEAREST)

    obj_A, mask_A = _rotate_pair(obj_A, mask_A, 90 + random.uniform(-8, 8))
    obj_B, mask_B = _rotate_pair(obj_B, mask_B, random.uniform(-8, 8))

    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255,255,255,255))
    mA_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
    mB_canvas = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)

    cx = CANVAS_SIZE//2 + random.randint(-15, 15)
    cy = CANVAS_SIZE//2 + random.randint(-15, 15)
    Ax = cx + random.randint(-18, 18) - obj_A.width//2
    Ay = cy + random.randint(-18, 18) - obj_A.height//2
    Bx = cx + random.randint(-18, 18) - obj_B.width//2
    By = cy + random.randint(-18, 18) - obj_B.height//2

    if random.random() < 0.5:
        canvas.paste(obj_A, (Ax, Ay), obj_A)
        mA_canvas.paste(mask_A, (Ax, Ay), mask_A)
        canvas.paste(obj_B, (Bx, By), obj_B)
        mB_canvas.paste(mask_B, (Bx, By), mask_B)
    else:
        canvas.paste(obj_B, (Bx, By), obj_B)
        mB_canvas.paste(mask_B, (Bx, By), mask_B)
        canvas.paste(obj_A, (Ax, Ay), obj_A)
        mA_canvas.paste(mask_A, (Ax, Ay), mask_A)

    arrA = np.array(mA_canvas) > 0
    arrB = np.array(mB_canvas) > 0
    arrC = arrA & arrB

    overlap = int(arrC.sum())
    if overlap < MIN_OVERLAP_PIXELS or overlap / min(arrA.sum(), arrB.sum()) > MAX_OVERLAP_RATIO:
        return None

    final_img = Image.new("RGB", canvas.size, (255,255,255))
    final_img.paste(canvas, mask=canvas.split()[3])
    mC_canvas = Image.fromarray(arrC.astype(np.uint8)*255, mode="L")
    return final_img, mA_canvas, mB_canvas, mC_canvas
