# ============================================================
# FILE: pipeline/buoc9_tong_hop.py
# CHỨC NĂNG: Bước 9 - Nối mạch dữ liệu từ Bước 1 đến Bước 8
# CẢI TIẾN: Lưu ảnh trung gian mỗi bước để báo cáo
# ============================================================
import os
import shutil
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from config.settings import RESULTS_DIR
from medical_pipeline.pipeline.buoc1_tang_cuong_anh_giam_mo import run_deblur
from medical_pipeline.pipeline.buoc2_tach_tung_cum_nhiem_sac_the import run_segmentation, stack_images
from medical_pipeline.pipeline.buoc3_main import run_overlap_separation
from medical_pipeline.pipeline.buoc4_duoi_thang_nhiem_sac_the import run_straightening
from medical_pipeline.pipeline.buoc5_phan_loai_nhiem_sac_theo_loai import run_classification_by_type
from medical_pipeline.pipeline.buoc7_ghep_cap_hoan_hao import run_perfect_pairing
from medical_pipeline.pipeline.buoc8_ve_karyogram import run_render_karyogram

def safe_imwrite(path, img):
    """Ghi ảnh an toàn (hỗ trợ Unicode path trên Windows)."""
    if img is None:
        return
    path = Path(path)
    if len(img.shape) == 3:
        # BGR → RGB → PIL
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    elif len(img.shape) == 2:
        pil_img = Image.fromarray(img)
    else:
        return
    pil_img.save(str(path))


def run_end_to_end_up_to_step7(input_image_path: str, sex: str = "XX"):
    print("=" * 60)
    print("🌟 CHẠY TOÀN BỘ PIPELINE ĐẾN BƯỚC 8 (KARYOGRAM)")
    print("=" * 60)
    input_path = Path(input_image_path)
    if not input_path.exists():
        print(f"❌ Không tìm thấy ảnh: {input_image_path}")
        return

    # === CHUẨN BỊ THƯ MỤC ===
    report_dir = RESULTS_DIR / f"{input_path.stem}_report"
    if report_dir.exists():
        shutil.rmtree(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Thư mục trung gian
    step2_dir = report_dir / "step2_clusters"
    step3_dir = report_dir / "step3_separated"
    step4_dir = report_dir / "step4_straightened"
    step5_dir = report_dir / "step5_classified"
    
    for d in [step2_dir, step3_dir, step4_dir, step5_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Đọc ảnh gốc (hỗ trợ Unicode path)
    img_data = np.fromfile(str(input_path), dtype=np.uint8)
    original_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    safe_imwrite(report_dir / "00_original.png", original_img)

    # =========================================================
    # BƯỚC 1: TĂNG CƯỜNG ẢNH, GIẢM MỜ
    # =========================================================
    print("\n--- BƯỚC 1: TĂNG CƯỜNG ẢNH, GIẢM MỜ ---")
    deblurred_bgr = run_deblur(original_img)
    safe_imwrite(report_dir / "01_deblurred.png", deblurred_bgr)
    print(f"  ✅ Đã lưu ảnh làm nét → 01_deblurred.png")

    # =========================================================
    # BƯỚC 2: TÁCH TỪNG CỤM NST
    # =========================================================
    print("\n--- BƯỚC 2: TÁCH TỪNG CỤM NHIỄM SẮC THỂ ---")
    cropped_images, heatmap, red_mask, bboxes = run_segmentation(deblurred_bgr)
    print(f"  ✅ Đã cắt được {len(cropped_images)} cụm NST.")
    
    # Lưu heatmap
    safe_imwrite(report_dir / "02_segmentation_heatmap.png", heatmap)
    
    # Lưu từng cụm riêng lẻ
    for idx, cluster_img in enumerate(cropped_images):
        safe_imwrite(step2_dir / f"cluster_{idx:03d}.png", cluster_img)
    
    # Lưu ảnh tổng hợp tất cả cụm
    if cropped_images:
        stacked = stack_images(cropped_images, cols=5)
        if stacked is not None:
            safe_imwrite(report_dir / "02_all_clusters.png", stacked)
    
    if not cropped_images:
        print("❌ Không phát hiện được cụm NST nào. Dừng pipeline.")
        return

    # =========================================================
    # BƯỚC 3: XỬ LÝ CỤM CHẠM, CHỒNG (U-Net + Concavity)
    # =========================================================
    print("\n--- BƯỚC 3: XỬ LÝ CỤM CHẠM, CHỒNG ---")
    
    # Lưu tạm các cụm ra thư mục để buoc3_main đọc
    temp_overlap_dir = RESULTS_DIR / "real_predictions" / "temp_overlap"
    separated_dir = RESULTS_DIR / "real_predictions" / "separated_chromosomes"
    if temp_overlap_dir.exists():
        shutil.rmtree(temp_overlap_dir)
    if separated_dir.exists():
        shutil.rmtree(separated_dir)
    temp_overlap_dir.mkdir(parents=True, exist_ok=True)
    separated_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, img_bgr in enumerate(cropped_images):
        stem_name = f"{input_path.stem}_part_{idx:03d}"
        safe_imwrite(temp_overlap_dir / f"{stem_name}.png", img_bgr)
        
    # Gọi U-Net để phân loại và tách
    final_separated = run_overlap_separation(temp_overlap_dir, output_dir=separated_dir)
    print(f"  ✅ Trích xuất được {len(final_separated)} NST đơn.")
    
    # Lưu từng NST đơn vào report
    for idx, chrom in enumerate(final_separated):
        safe_imwrite(step3_dir / f"nst_{idx:03d}_{chrom['stem']}.png", chrom['image'])
    
    # Lưu ảnh tổng hợp NST đơn
    if final_separated:
        single_imgs = [c['image'] for c in final_separated if c['image'] is not None]
        # Resize tất cả về cùng kích thước để stack
        resized_singles = []
        for img in single_imgs:
            if img is not None:
                r = cv2.resize(img, (128, 128))
                resized_singles.append(r)
        if resized_singles:
            stacked_singles = stack_images(resized_singles, cols=8)
            if stacked_singles is not None:
                safe_imwrite(report_dir / "03_all_separated.png", stacked_singles)
    
    if not final_separated:
        print("❌ Không tách được NST đơn nào. Dừng pipeline.")
        return

    # =========================================================
    # BƯỚC 4: DUỖI THẲNG NHIỄM SẮC THỂ
    # =========================================================
    print("\n--- BƯỚC 4: DUỖI THẲNG NHIỄM SẮC THỂ ---")
    # Chuẩn bị dữ liệu: đảm bảo mỗi chromosome dict có đủ image + mask
    chromosomes = final_separated
    chromosomes = run_straightening(chromosomes, report_dir)
    print(f"  ✅ Đã duỗi thẳng {len(chromosomes)} NST.")
    
    # Lưu ảnh tổng hợp NST đã duỗi
    straight_imgs = []
    for c in chromosomes:
        img = c.get("straight_image", c.get("aligned_image", c.get("image")))
        if img is not None:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            straight_imgs.append(cv2.resize(img, (128, 128)))
    if straight_imgs:
        stacked_straight = stack_images(straight_imgs, cols=8)
        if stacked_straight is not None:
            safe_imwrite(report_dir / "04_all_straightened.png", stacked_straight)

    # =========================================================
    # BƯỚC 5: PHÂN LOẠI NST THEO LOẠI
    # =========================================================
    print("\n--- BƯỚC 5: PHÂN LOẠI NHIỄM SẮC THEO LOẠI ---")
    chromosomes = run_classification_by_type(chromosomes)
    
    # In kết quả phân loại
    print(f"\n  Kết quả phân loại {len(chromosomes)} NST:")
    for c in chromosomes:
        label = c.get("type_label", c.get("label", "?"))
        stem = c.get("stem", "?")
        print(f"    {stem} → NST {label}")
    
    # Lưu ảnh với nhãn phân loại
    for c in chromosomes:
        img = c.get("straight_image", c.get("aligned_image", c.get("image")))
        label = c.get("type_label", c.get("label", "?"))
        if img is not None:
            # Tạo ảnh có nhãn
            if len(img.shape) == 2:
                labeled_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                labeled_img = img.copy()
            cv2.putText(labeled_img, f"chr{label}", (5, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            safe_imwrite(step5_dir / f"{c.get('stem','?')}_chr{label}.png", labeled_img)

    # =========================================================
    # BƯỚC 7: GHÉP CẶP HOÀN HẢO
    # =========================================================
    pairing = run_perfect_pairing(chromosomes, sex)

    # =========================================================
    # BƯỚC 8: VẼ KARYOGRAM
    # =========================================================
    karyogram_output = report_dir / f"08_final_karyogram_{sex}.png"
    run_render_karyogram(chromosomes, pairing, sex, karyogram_output)

    print("\n🎉 ĐÃ CHẠY HOÀN TOÀN TẤT CẢ CÁC BƯỚC VÀ VẼ KARYOGRAM!")
    print(f"📁 Kết quả lưu tại: {report_dir}")
    
    return report_dir
