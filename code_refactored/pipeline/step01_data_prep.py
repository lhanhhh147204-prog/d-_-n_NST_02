# ============================================================
# FILE: pipeline/01_data_prep.py
# CHỨC NĂNG: Tích hợp Toàn Bộ Luồng Xử Lý Dữ Liệu (Bước 1 -> 5)
# ============================================================

import os
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

from config.settings import (
    RAW_SINGLE_DIR, RAW_OVERLAP_DIR, PREP_SINGLE_DIR, GEN_DATA_DIR, 
    PROCESSED_256_DIR, DATASET_DIR, RESULTS_DIR, IMG_SIZE, BG_MARGIN, 
    CROP_PADDING, KERNEL_SIZE, MASK_SHRINK_ITER, CANVAS_SIZE, 
    TARGET_LONG_SIDE_MIN, TARGET_LONG_SIDE_MAX, MIN_OVERLAP_PIXELS, 
    MAX_OVERLAP_RATIO, TOTAL_SAMPLES_TO_GENERATE, SPLIT_SEED, 
    TRAIN_RATIO, VAL_RATIO
)
from core.image_utils import get_border_background, keep_largest_components, resize_with_padding, binarize_mask

def step_1_create_folders():
    """Khởi tạo toàn bộ cây thư mục cho dự án."""
    print("--- BƯỚC 1: KHỞI TẠO THƯ MỤC ---")
    folders = [
        RAW_SINGLE_DIR, RAW_OVERLAP_DIR,
        PREP_SINGLE_DIR / "images_rgba", PREP_SINGLE_DIR / "masks",
        GEN_DATA_DIR / "images", GEN_DATA_DIR / "masks_A", GEN_DATA_DIR / "masks_B", GEN_DATA_DIR / "masks_C",
        PROCESSED_256_DIR / "images", PROCESSED_256_DIR / "masks_A", PROCESSED_256_DIR / "masks_B", PROCESSED_256_DIR / "masks_C",
    ]
    for split in ["train", "val", "test"]:
        folders.extend([
            DATASET_DIR / split / "images", DATASET_DIR / split / "masks_A",
            DATASET_DIR / split / "masks_B", DATASET_DIR / split / "masks_C"
        ])
    for f in folders:
        f.mkdir(parents=True, exist_ok=True)
    print("✅ Đã tạo cấu trúc thư mục.")

def extract_object_rgba(img_path: Path) -> tuple:
    """Tách NST đơn ra khỏi nền trắng."""
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
    if len(ys) == 0: return None, None
    x1, x2 = max(0, xs.min()-CROP_PADDING), min(img.width, xs.max()+CROP_PADDING)
    y1, y2 = max(0, ys.min()-CROP_PADDING), min(img.height, ys.max()+CROP_PADDING)
    
    cropped_mask = object_mask[y1:y2, x1:x2]
    cropped_img = np.array(img)[y1:y2, x1:x2]
    
    rgba = np.zeros((cropped_img.shape[0], cropped_img.shape[1], 4), dtype=np.uint8)
    rgba[..., :3] = cropped_img
    rgba[..., 3] = cropped_mask.astype(np.uint8) * 255
    
    return Image.fromarray(rgba, mode="RGBA"), Image.fromarray(cropped_mask.astype(np.uint8)*255, mode="L")

def step_2_prepare_single():
    print("--- BƯỚC 2: TÁCH NST ĐƠN KHỎI NỀN ---")
    image_paths = list(RAW_SINGLE_DIR.glob("*.png")) + list(RAW_SINGLE_DIR.glob("*.jpg"))
    out_img = PREP_SINGLE_DIR / "images_rgba"
    out_mask = PREP_SINGLE_DIR / "masks"
    if not image_paths:
        print("⚠️ Không tìm thấy ảnh NST đơn lẻ nào.")
        return
    for img_path in tqdm(image_paths, desc="Xử lý ảnh đơn"):
        rgba, mask = extract_object_rgba(img_path)
        if rgba is not None:
            rgba.save(out_img / f"{img_path.stem}.png")
            mask.save(out_mask / f"{img_path.stem}.png")
    print("✅ Hoàn thành tách NST đơn.")

def _rotate_pair(img: Image.Image, mask: Image.Image, angle: float) -> tuple:
    return (
        img.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(255,255,255,0)),
        mask.rotate(angle, resample=Image.NEAREST, expand=True, fillcolor=0)
    )

def make_overlap_canvas(obj_A, mask_A, obj_B, mask_B):
    """Ghép 2 NST tạo vùng overlap (C = A ∩ B)."""
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
    
    cx, cy = CANVAS_SIZE//2 + random.randint(-15,15), CANVAS_SIZE//2 + random.randint(-15,15)
    Ax, Ay = cx + random.randint(-18,18) - obj_A.width//2, cy + random.randint(-18,18) - obj_A.height//2
    Bx, By = cx + random.randint(-18,18) - obj_B.width//2, cy + random.randint(-18,18) - obj_B.height//2
    
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

def step_3_generate_synthetic():
    print("--- BƯỚC 3: SINH DATA TỔNG HỢP ---")
    imgs = list((PREP_SINGLE_DIR / "images_rgba").glob("*.png"))
    masks = list((PREP_SINGLE_DIR / "masks").glob("*.png"))
    if not imgs: return
    
    data = []
    for i, m in zip(imgs, masks):
        data.append((Image.open(i).convert("RGBA"), Image.open(m).convert("L")))
        
    count = 0
    with tqdm(total=TOTAL_SAMPLES_TO_GENERATE, desc="Sinh ảnh chồng nhau") as pbar:
        while count < TOTAL_SAMPLES_TO_GENERATE:
            A = random.choice(data)
            B = random.choice(data)
            res = make_overlap_canvas(A[0], A[1], B[0], B[1])
            if res:
                img, ma, mb, mc = res
                name = f"img_{count+1:06d}.png"
                img.save(GEN_DATA_DIR / "images" / name)
                ma.save(GEN_DATA_DIR / "masks_A" / name)
                mb.save(GEN_DATA_DIR / "masks_B" / name)
                mc.save(GEN_DATA_DIR / "masks_C" / name)
                count += 1
                pbar.update(1)
    print("✅ Hoàn thành sinh data tổng hợp.")

def step_4_preprocess_to_256():
    print("--- BƯỚC 4: CHUẨN HÓA VỀ 256x256 ---")
    images = list((GEN_DATA_DIR / "images").glob("*.png"))
    for p in tqdm(images, desc="Resize & Pad"):
        name = p.name
        img = Image.open(p)
        ma = Image.open(GEN_DATA_DIR / "masks_A" / name)
        mb = Image.open(GEN_DATA_DIR / "masks_B" / name)
        mc = Image.open(GEN_DATA_DIR / "masks_C" / name)
        
        img_res, _ = resize_with_padding(img, IMG_SIZE, is_mask=False)
        ma_res, _ = resize_with_padding(ma, IMG_SIZE, is_mask=True)
        mb_res, _ = resize_with_padding(mb, IMG_SIZE, is_mask=True)
        mc_res, _ = resize_with_padding(mc, IMG_SIZE, is_mask=True)
        
        img_res.save(PROCESSED_256_DIR / "images" / name)
        binarize_mask(ma_res).save(PROCESSED_256_DIR / "masks_A" / name)
        binarize_mask(mb_res).save(PROCESSED_256_DIR / "masks_B" / name)
        binarize_mask(mc_res).save(PROCESSED_256_DIR / "masks_C" / name)
    print("✅ Hoàn thành chuẩn hóa 256x256.")

def step_5_split_data():
    print("--- BƯỚC 5: CHIA DATASET (TRAIN/VAL/TEST) ---")
    files = [f.name for f in (PROCESSED_256_DIR / "images").glob("*.png")]
    random.seed(SPLIT_SEED)
    random.shuffle(files)
    
    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    
    splits = {
        "train": files[:n_train],
        "val": files[n_train:n_train+n_val],
        "test": files[n_train+n_val:]
    }
    
    for split, names in splits.items():
        print(f"Copying {len(names)} files for {split}...")
        for name in names:
            shutil.copy(PROCESSED_256_DIR / "images" / name, DATASET_DIR / split / "images" / name)
            shutil.copy(PROCESSED_256_DIR / "masks_A" / name, DATASET_DIR / split / "masks_A" / name)
            shutil.copy(PROCESSED_256_DIR / "masks_B" / name, DATASET_DIR / split / "masks_B" / name)
            shutil.copy(PROCESSED_256_DIR / "masks_C" / name, DATASET_DIR / split / "masks_C" / name)
    print("✅ Hoàn thành chia dataset.")

def run_all():
    step_1_create_folders()
    step_2_prepare_single()
    step_3_generate_synthetic()
    step_4_preprocess_to_256()
    step_5_split_data()
    print("🎉 QUÁ TRÌNH CHUẨN BỊ DỮ LIỆU ĐÃ HOÀN TẤT!")

if __name__ == "__main__":
    run_all()
