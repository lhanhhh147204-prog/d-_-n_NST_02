# ============================================================
# FILE: karyogram/builder.py
# CHỨC NĂNG: Render ảnh Karyogram chuẩn lâm sàng y khoa
# THAY THẾ: mock_karyotype.py (phiên bản thô sơ)
# GHI CHÚ: Output là ảnh độ phân giải cao, xếp theo chuẩn Denver
# ============================================================

"""
Module Vẽ Karyogram (Clinical-Grade Karyogram Builder).

Karyogram = Bảng xếp đôi 23 cặp NST theo thứ tự kích thước giảm dần,
chia theo 7 nhóm Denver (A-G) — chuẩn quốc tế trong di truyền học y tế.

Cấu trúc Karyogram chuẩn:
┌─────────────────────────────────────────────────┐
│   Nhóm A (1-3)  │  Nhóm B (4-5)               │
│─────────────────│───────────────────────────────│
│   Nhóm C (6-12, X)                             │
│─────────────────────────────────────────────────│
│   Nhóm D (13-15) │ Nhóm E (16-18)             │
│──────────────────│──────────────────────────────│
│   Nhóm F (19-20) │ Nhóm G (21-22, Y)          │
└─────────────────────────────────────────────────┘

Input: Dict {label → [ảnh NST đã aligned]}
Output: Ảnh PNG/TIFF độ phân giải cao

LƯU Ý:
- Ảnh NST phải đã được xoay thẳng đứng + p-arm lên trên
- Module này chỉ vẽ, KHÔNG xử lý hình thái
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from config.settings import DENVER_GROUPS, KARYOTYPE_LABELS, KARYOGRAM_OUTPUT_DIR


# =========================================================
# HẰNG SỐ RENDER
# =========================================================

# Kích thước mỗi ô NST (pixel)
CELL_WIDTH = 90
CELL_HEIGHT = 180

# Khoảng cách giữa 2 NST trong 1 cặp
PAIR_GAP = 8

# Khoảng cách giữa các cặp
PAIR_SPACING = 30

# Khoảng cách giữa các hàng (nhóm Denver)
ROW_SPACING = 60

# Padding viền ngoài
MARGIN = 40

# Chiều cao khu vực nhãn
LABEL_HEIGHT = 30

# Màu sắc
BG_COLOR = (255, 255, 255)       # Nền trắng
BORDER_COLOR = (180, 180, 180)    # Viền xám nhạt
LABEL_COLOR = (60, 60, 60)        # Chữ nhãn xám đậm
GROUP_LABEL_COLOR = (30, 100, 180)  # Chữ nhóm Denver — xanh dương
SEPARATOR_COLOR = (200, 200, 200)   # Đường phân cách nhóm


def _resize_chromosome(
    img: np.ndarray,
    target_height: int = CELL_HEIGHT,
) -> np.ndarray:
    """
    Resize ảnh NST: CHỈ thu nhỏ nếu quá to, KHÔNG phóng to các NST nhỏ.
    Giữ nguyên tỷ lệ sinh học (relative size) giữa các NST.
    """
    if img is None or img.size == 0:
        return np.full((target_height, CELL_WIDTH), 255, dtype=np.uint8)

    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.full((target_height, CELL_WIDTH), 255, dtype=np.uint8)

    # Chỉ thu nhỏ nếu ảnh gốc lớn hơn khung (tránh phóng to NST số 22 bằng NST số 1)
    scale = min(1.0, target_height / h, CELL_WIDTH / w)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))

    # Chuyển về 2D nếu cần
    if len(img.shape) == 3:
        img_2d = img[:, :, 0] if img.shape[2] >= 1 else img.mean(axis=2)
    else:
        img_2d = img

    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(img_2d.astype(np.uint8))
    try:
        resample_filter = PILImage.Resampling.LANCZOS
    except AttributeError:
        resample_filter = PILImage.LANCZOS
        
    if scale < 1.0:
        pil_resized = pil_img.resize((new_w, new_h), resample_filter)
    else:
        pil_resized = pil_img

    # Tạo canvas trắng và paste ảnh vào giữa (canh dưới để các NST đứng trên cùng đường thẳng)
    canvas = np.full((target_height, CELL_WIDTH), 255, dtype=np.uint8)
    offset_x = (CELL_WIDTH - new_w) // 2
    offset_y = target_height - new_h - 10  # Canh lề dưới (bottom alignment) chừa 10px
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = np.array(pil_resized)

    return canvas


def _try_load_font(size: int = 14):
    """Thử load font hệ thống, fallback về font mặc định."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except (OSError, IOError):
            return ImageFont.load_default()


def build_karyogram(
    chromosome_images: Dict[str, List[np.ndarray]],
    sex: str = "XX",
    title: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> Image.Image:
    """
    Vẽ Karyogram chuẩn lâm sàng từ dict ảnh NST đã phân loại.

    Tham số:
        chromosome_images: Dict {label → [ảnh_1, ảnh_2]}.
            - label: "1", "2", ..., "22", "X", "Y"
            - Mỗi ảnh là numpy (H, W), uint8 grayscale
            - Danh sách có thể có 0, 1, hoặc 2 ảnh
        sex: "XX" hoặc "XY" — hiển thị trên tiêu đề.
        title: Tiêu đề tùy chỉnh (nếu None → dùng mặc định).
        output_path: Đường dẫn lưu file (nếu None → chỉ trả về Image).

    Trả về:
        PIL Image object — ảnh Karyogram hoàn chỉnh.
    """
    # Tính kích thước canvas
    # Đếm số cặp trên mỗi hàng (mỗi nhóm Denver = 1 hàng)
    row_configs = []
    for group_name, labels in DENVER_GROUPS.items():
        num_pairs = len(labels)
        row_width = num_pairs * (2 * CELL_WIDTH + PAIR_GAP + PAIR_SPACING)
        row_configs.append((group_name, labels, row_width))

    max_row_width = max(rc[2] for rc in row_configs)
    canvas_width = max_row_width + 2 * MARGIN
    canvas_height = (
        MARGIN  # Trên
        + 50  # Tiêu đề
        + len(row_configs) * (CELL_HEIGHT + LABEL_HEIGHT + ROW_SPACING)
        + MARGIN  # Dưới
    )

    # Tạo canvas
    canvas = Image.new("RGB", (canvas_width, canvas_height), BG_COLOR)
    draw = ImageDraw.Draw(canvas)

    # Load font
    title_font = _try_load_font(20)
    label_font = _try_load_font(13)
    group_font = _try_load_font(16)

    # Vẽ tiêu đề
    if title is None:
        title = f"KARYOGRAM — {sex}"
    # Dùng textbbox thay cho textsize (Pillow >= 10)
    try:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
    except AttributeError:
        tw = len(title) * 12  # Fallback
    draw.text(((canvas_width - tw) // 2, MARGIN), title, fill=LABEL_COLOR, font=title_font)

    # Vẽ từng hàng (nhóm Denver)
    current_y = MARGIN + 50

    for group_name, labels, _ in row_configs:
        current_x = MARGIN  # Bắt đầu vẽ cặp NST

        for label in labels:
            images = chromosome_images.get(label, [])

            # Lấy tối đa 2 ảnh cho cặp
            img_1 = _resize_chromosome(images[0]) if len(images) > 0 else None
            img_2 = _resize_chromosome(images[1]) if len(images) > 1 else None

            # Vẽ NST thứ 1 (hoặc ô trống)
            if img_1 is not None:
                pil_1 = Image.fromarray(img_1)
                canvas.paste(pil_1, (current_x, current_y))
            else:
                # Ô trống — viền chấm
                draw.rectangle(
                    [current_x, current_y, current_x + CELL_WIDTH, current_y + CELL_HEIGHT],
                    outline=BORDER_COLOR,
                )
                draw.text(
                    (current_x + CELL_WIDTH // 2 - 5, current_y + CELL_HEIGHT // 2 - 5),
                    "?", fill=BORDER_COLOR, font=label_font,
                )

            # Vẽ NST thứ 2
            x2 = current_x + CELL_WIDTH + PAIR_GAP
            if img_2 is not None:
                pil_2 = Image.fromarray(img_2)
                canvas.paste(pil_2, (x2, current_y))
            else:
                if label != "Y" or sex == "XY":
                    draw.rectangle(
                        [x2, current_y, x2 + CELL_WIDTH, current_y + CELL_HEIGHT],
                        outline=BORDER_COLOR,
                    )
                    draw.text(
                        (x2 + CELL_WIDTH // 2 - 5, current_y + CELL_HEIGHT // 2 - 5),
                        "?", fill=BORDER_COLOR, font=label_font,
                    )

            # Nhãn dưới cặp
            label_x = current_x + (2 * CELL_WIDTH + PAIR_GAP) // 2
            draw.text(
                (label_x - 10, current_y + CELL_HEIGHT + 5),
                f"Chr {label}",
                fill=LABEL_COLOR,
                font=label_font,
            )

            current_x += 2 * CELL_WIDTH + PAIR_GAP + PAIR_SPACING

        current_y += CELL_HEIGHT + LABEL_HEIGHT + ROW_SPACING

    # Lưu file nếu có output_path
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(str(output_path), quality=95)
        print(f"✅ [Karyogram] Đã lưu ảnh Karyogram tại: {output_path}")

    return canvas


def build_karyogram_from_files(
    source_dir: Path,
    sex: str = "XX",
    output_path: Optional[Path] = None,
) -> Image.Image:
    """
    Vẽ Karyogram từ thư mục chứa ảnh NST đã phân loại.

    Cấu trúc thư mục mong đợi:
        source_dir/
        ├── 1/   ← chứa ảnh NST loại 1
        ├── 2/   ← chứa ảnh NST loại 2
        ├── ...
        ├── X/
        └── Y/

    Hoặc dạng flat:
        source_dir/
        ├── chromosome_1_A.png
        ├── chromosome_1_B.png
        ├── ...
    """
    import glob

    chromosome_images: Dict[str, List[np.ndarray]] = {label: [] for label in KARYOTYPE_LABELS}

    # Thử cấu trúc thư mục trước
    for label in KARYOTYPE_LABELS:
        label_dir = source_dir / label
        if label_dir.is_dir():
            for img_path in sorted(label_dir.glob("*.png"))[:2]:
                img = np.array(Image.open(img_path).convert("L"))
                chromosome_images[label].append(img)

    # Nếu không có thư mục con → thử dạng flat
    total = sum(len(v) for v in chromosome_images.values())
    if total == 0:
        all_files = sorted(source_dir.glob("*.png"))
        for img_path in all_files:
            name_lower = img_path.stem.lower()
            for label in KARYOTYPE_LABELS:
                pattern = f"_chromosome_{label.lower()}_"
                if pattern in name_lower or name_lower.endswith(f"_{label.lower()}"):
                    if len(chromosome_images[label]) < 2:
                        img = np.array(Image.open(img_path).convert("L"))
                        chromosome_images[label].append(img)
                    break

    return build_karyogram(chromosome_images, sex=sex, output_path=output_path)
