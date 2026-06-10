# ============================================================
# FILE: core/metrics.py
# CHỨC NĂNG: Các hàm tính toán số liệu đánh giá (NumPy)
# ============================================================

import numpy as np

def calculate_dice(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính hệ số Dice = 2 * |A ∩ B| / (|A| + |B|).
    Nếu cả 2 đều rỗng (0), trả về 1.0 (dự đoán đúng là không có gì).
    """
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    intersection = np.logical_and(y_true_bool, y_pred_bool).sum()
    total = y_true_bool.sum() + y_pred_bool.sum()
    if total == 0:
        return 1.0
    return 2.0 * intersection / total

def calculate_iou(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Tính Intersection over Union (IoU) = |A ∩ B| / |A ∪ B|.
    Nếu cả 2 đều rỗng (0), trả về 1.0.
    """
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    intersection = np.logical_and(y_true_bool, y_pred_bool).sum()
    union = np.logical_or(y_true_bool, y_pred_bool).sum()
    if union == 0:
        return 1.0
    return intersection / union

def calculate_pixel_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính tỉ lệ số pixel được dự đoán đúng trên tổng số pixel."""
    y_true_bool = y_true.astype(bool)
    y_pred_bool = y_pred.astype(bool)
    correct = (y_true_bool == y_pred_bool).sum()
    total = y_true_bool.size
    if total == 0:
        return 0.0
    return correct / total
