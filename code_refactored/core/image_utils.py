# ============================================================
# FILE: core/image_utils.py
# CHỨC NĂNG: Các hàm tiện ích xử lý ảnh (tiền xử lý, hậu xử lý)
# ============================================================

from typing import Dict, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw

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

def resize_with_padding(img: Image.Image, target_size: int = 256, is_mask: bool = False) -> Tuple[Image.Image, Dict[str, int]]:
    """
    Resize giữ tỉ lệ, thêm padding để đạt kích thước target_size x target_size.
    Trả về ảnh (hoặc mask) và dictionary chứa thông tin để restore về sau.
    """
    if not is_mask:
        img = img.convert("L")
    w, h = img.size
    scale = min(target_size / w, target_size / h)
    
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    
    resample = Image.NEAREST if is_mask else Image.BILINEAR
    fill_color = 0 if is_mask else 255
    
    img_resized = img.resize((new_w, new_h), resample)
    canvas = Image.new("L", (target_size, target_size), fill_color)
    
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(img_resized, (paste_x, paste_y))
    
    meta = {
        "orig_w": w, "orig_h": h,
        "new_w": new_w, "new_h": new_h,
        "paste_x": paste_x, "paste_y": paste_y,
    }
    return canvas, meta

def restore_mask_to_original(mask_256: np.ndarray, meta: Dict[str, int]) -> np.ndarray:
    """Đảo ngược bước padding, map mask 256x256 về kích thước gốc."""
    x, y = meta["paste_x"], meta["paste_y"]
    nw, nh = meta["new_w"], meta["new_h"]
    ow, oh = meta["orig_w"], meta["orig_h"]
    
    crop = mask_256[y:y+nh, x:x+nw].astype(np.uint8) * 255
    restored = Image.fromarray(crop, mode="L").resize((ow, oh), Image.NEAREST)
    return np.array(restored) > 127

def inpaint_instance(gray: np.ndarray, instance_mask: np.ndarray, overlap_mask: np.ndarray, radius: int = 3) -> np.ndarray:
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

def rgba_from_instance(gray: np.ndarray, instance_mask: np.ndarray, overlap_mask: np.ndarray, transparent: bool = True) -> Image.Image:
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

def mask_to_contour(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(mask_uint8)
    cv2.drawContours(contour_img, contours, -1, 255, thickness=1)
    return contour_img

def make_overlay(base_img: Image.Image, mask_A: np.ndarray, mask_B: np.ndarray, mask_C: np.ndarray) -> Image.Image:
    base = np.array(base_img.convert("RGB")).astype(np.float32)
    overlay = base.copy()
    
    A, B, C = mask_A.astype(bool), mask_B.astype(bool), mask_C.astype(bool)
    overlay[A] = overlay[A] * 0.4 + np.array([255, 0, 0]) * 0.6
    overlay[B] = overlay[B] * 0.4 + np.array([0, 255, 0]) * 0.6
    overlay[C] = overlay[C] * 0.3 + np.array([255, 255, 0]) * 0.7
    
    return Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

def binarize_mask(mask_img: Image.Image) -> Image.Image:
    """Chuyển đổi mask thành dạng nhị phân thuần túy 0 và 255."""
    arr = np.array(mask_img)
    arr = (arr > 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")
