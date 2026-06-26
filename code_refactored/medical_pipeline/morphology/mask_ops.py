# ============================================================
# FILE: morphology/mask_ops.py
# CHỨC NĂNG: Các thao tác xử lý mask hình thái học cho NST
# TÁCH TỪ: core/image_utils.py (phần phân tích cấu trúc mask)
# ============================================================

from typing import Tuple
import cv2
import numpy as np


def keep_largest_components(mask: np.ndarray, keep: int = 1) -> np.ndarray:
    """Giữ component lớn nhất để giảm noise; keep=0 thì bỏ qua."""
    if keep <= 0:
        return mask.astype(bool)
    mask_u8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    order = np.argsort(areas)[::-1][:keep] + 1
    return np.isin(labels, order)


def clean_mask(mask: np.ndarray, keep_components: int = 1) -> np.ndarray:
    """Làm sạch mask bằng morphology open/close và giữ component lớn nhất."""
    mask_uint8 = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    cleaned = keep_largest_components(cleaned > 0, keep=keep_components)
    return cleaned.astype(bool)


def get_border_background(gray: np.ndarray, bg_margin: int = 15) -> Tuple[np.ndarray, int, float]:
    """Phát hiện nền bằng cách tìm vùng sáng tiếp xúc với biên ảnh."""
    h, w = gray.shape
    border_pixels = np.concatenate([
        gray[0, :], gray[h-1, :],
        gray[:, 0], gray[:, w-1]
    ])
    bg_ref = float(np.median(border_pixels))
    threshold = max(210, int(bg_ref - bg_margin))

    candidate_bg = gray >= threshold
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_bg.astype(np.uint8), connectivity=8
    )
    border_labels = set(np.unique(np.concatenate([
        labels[0, :], labels[h-1, :],
        labels[:, 0], labels[:, w-1]
    ])))
    border_labels.discard(0)

    background = np.isin(labels, list(border_labels)) & candidate_bg
    return background, threshold, bg_ref
