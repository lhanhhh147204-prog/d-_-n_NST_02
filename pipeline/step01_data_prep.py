# ============================================================
# FILE: pipeline/step01_data_prep.py
# CHỨC NĂNG: Controller — điều phối Toàn Bộ Luồng Chuẩn Bị Dữ Liệu
# Logic nghiệp vụ nằm trong: data/preprocessing.py, utils/image_io.py
# ============================================================

import random
import shutil
from pathlib import Path
from tqdm import tqdm
from PIL import Image

from config.settings import (
    RAW_SINGLE_DIR, RAW_OVERLAP_DIR, PREP_SINGLE_DIR, GEN_DATA_DIR,
    PROCESSED_256_DIR, DATASET_DIR, RESULTS_DIR, IMG_SIZE,
    TOTAL_SAMPLES_TO_GENERATE, SPLIT_SEED, TRAIN_RATIO, VAL_RATIO,
)
from data.preprocessing import extract_object_rgba, make_overlap_canvas
from utils.image_io import resize_with_padding, binarize_mask


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


def step_2_prepare_single():
    """Tách từng NST đơn ra khỏi nền trắng (dùng data.preprocessing)."""
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


def step_3_generate_synthetic():
    """Sinh dữ liệu tổng hợp NST chồng nhau (dùng data.preprocessing)."""
    print("--- BƯỚC 3: SINH DATA TỔNG HỢP ---")
    imgs = list((PREP_SINGLE_DIR / "images_rgba").glob("*.png"))
    masks = list((PREP_SINGLE_DIR / "masks").glob("*.png"))
    if not imgs:
        return

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
    """Chuẩn hóa toàn bộ ảnh về 256x256 (dùng utils.image_io)."""
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
    """Chia dataset thành Train/Val/Test theo tỉ lệ cấu hình."""
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
