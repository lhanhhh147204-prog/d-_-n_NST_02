# ============================================================
# FILE: utils/image_io.py
# CHỨC NĂNG: Công cụ đọc/ghi ảnh, resize, restore, chuẩn hóa
# TÁCH TỪ: core/image_utils.py (phần xử lý I/O và biến đổi ảnh)
# ============================================================

from pathlib import Path
from typing import Dict, Tuple
import numpy as np
from PIL import Image


def resize_with_padding(
    img: Image.Image,
    target_size: int = 256,
    is_mask: bool = False
) -> Tuple[Image.Image, Dict[str, int]]:
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


def binarize_mask(mask_img: Image.Image) -> Image.Image:
    """Chuyển đổi mask thành dạng nhị phân thuần túy 0 và 255."""
    arr = np.array(mask_img)
    arr = (arr > 127).astype(np.uint8) * 255
    return Image.fromarray(arr, mode="L")


def prepare_input(img: Image.Image) -> np.ndarray:
    """Chuẩn bị ảnh PIL thành numpy array 4D (batch=1) để đưa vào model."""
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=-1)
    return np.expand_dims(arr, axis=0)


def save_binary_mask(mask: np.ndarray, path: Path):
    """Lưu mask numpy boolean/uint8 thành file ảnh PNG."""
    mask_img = (mask.astype(np.uint8) * 255)
    Image.fromarray(mask_img, mode="L").save(path)
