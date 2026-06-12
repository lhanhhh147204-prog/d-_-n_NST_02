# ============================================================
# FILE: morphology/inpainting.py
# CHỨC NĂNG: Tách NST chồng lấn bằng inpainting TELEA
# TÁCH TỪ: core/image_utils.py (phần inpainting và tách NST)
# ============================================================

import cv2
import numpy as np
from PIL import Image


def inpaint_instance(
    gray: np.ndarray,
    instance_mask: np.ndarray,
    overlap_mask: np.ndarray,
    radius: int = 3
) -> np.ndarray:
    """Fill vùng bị che khuất (C) bằng inpainting TELEA."""
    gray_u8 = gray.astype(np.uint8)
    work = gray_u8.copy()
    work[~instance_mask] = 255  # Set nền trắng ngoài vùng mask

    fill_region = (instance_mask & overlap_mask).astype(np.uint8) * 255
    if fill_region.sum() == 0:
        return work

    kernel = np.ones((3, 3), np.uint8)
    fill_region = cv2.dilate(fill_region, kernel, iterations=1)
    fill_region[~instance_mask] = 0

    return cv2.inpaint(work, fill_region, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)


def rgba_from_instance(
    gray: np.ndarray,
    instance_mask: np.ndarray,
    overlap_mask: np.ndarray,
    transparent: bool = True
) -> Image.Image:
    """Chuyển đổi vùng mask inpaint thành ảnh RGBA nền trong suốt (hoặc RGB nền trắng)."""
    filled = inpaint_instance(gray, instance_mask, overlap_mask)
    if transparent:
        rgba = np.zeros((gray.shape[0], gray.shape[1], 4), dtype=np.uint8)
        rgba[..., 0:3] = np.stack([filled]*3, axis=-1)
        rgba[..., 3] = instance_mask.astype(np.uint8) * 255
        return Image.fromarray(rgba, mode="RGBA")

    out = np.full_like(gray, 255, dtype=np.uint8)
    out[instance_mask] = filled[instance_mask]
    return Image.fromarray(out, mode="L")
