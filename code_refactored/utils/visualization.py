# ============================================================
# FILE: utils/visualization.py
# CHỨC NĂNG: Công cụ visualize kết quả (contour, overlay màu)
# TÁCH TỪ: core/image_utils.py (phần visualize/reporting)
# ============================================================

import cv2
import numpy as np
from PIL import Image


def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    """Trích xuất đường biên contour từ mask boolean/uint8."""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(mask_uint8)
    cv2.drawContours(contour_img, contours, -1, 255, thickness=1)
    return contour_img


def make_overlay(
    base_img: Image.Image,
    mask_A: np.ndarray,
    mask_B: np.ndarray,
    mask_C: np.ndarray
) -> Image.Image:
    """
    Đè màu mask A (đỏ), B (xanh lá), C (vàng) lên ảnh gốc để xem kết quả.
    - Mask A (NST thứ nhất): màu đỏ
    - Mask B (NST thứ hai): màu xanh lá
    - Mask C (vùng giao chồng): màu vàng
    """
    base = np.array(base_img.convert("RGB")).astype(np.float32)
    overlay = base.copy()

    A, B, C = mask_A.astype(bool), mask_B.astype(bool), mask_C.astype(bool)
    overlay[A] = overlay[A] * 0.4 + np.array([255, 0, 0]) * 0.6
    overlay[B] = overlay[B] * 0.4 + np.array([0, 255, 0]) * 0.6
    overlay[C] = overlay[C] * 0.3 + np.array([255, 255, 0]) * 0.7

    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))
