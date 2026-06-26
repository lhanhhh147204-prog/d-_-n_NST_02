# ==========================================
# BƯỚC 8: VẼ KARYOGRAM
# CHỨC NĂNG: Giao tiếp với module karyogram/builder.py
# ==========================================

from medical_pipeline.karyogram.builder import build_karyogram
from pathlib import Path

def run_render_karyogram(chromosomes, pairing, sex, output_path: Path):
    """
    chromosomes: list of dict, mỗi dict có 'label' (lớp phân loại 1-22, X, Y), 'image' (ảnh)
    pairing: dict map id NST -> cặp. Thực tế buoc7 không thay đổi cấu trúc chromosome nhiều.
    Để dễ dàng, hàm này chuyển đổi chromosomes thành định dạng dict {label -> [img1, img2]} cho builder.
    """
    
    # Gom nhóm ảnh theo label
    chr_dict = {}
    for chrom in chromosomes:
        label = chrom.get("type_label") or chrom.get("label")
        if not label:
            continue
            
        label_str = str(label)
        if label_str not in chr_dict:
            chr_dict[label_str] = []
            
        img = chrom.get("straight_image")
        if img is None:
            img = chrom.get("image")
            
        chr_dict[label_str].append(img)
        
    print(f"\n[BƯỚC 8] Tiến hành vẽ Karyogram ({sex})...")
    build_karyogram(chr_dict, sex=sex, output_path=output_path)
    
    # === CẢI TIẾN BƯỚC 8: CẢNH BÁO LÂM SÀNG ===
    total_chroms = sum(len(v) for v in pairing.values())
    if total_chroms < 44 or total_chroms > 48:
        import cv2
        import numpy as np
        
        # Đọc ảnh có đường dẫn Unicode (Tiếng Việt)
        img_array = np.fromfile(str(output_path), np.uint8)
        final_karyogram = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if final_karyogram is not None:
            warning_text = f"WARNING: ABNORMAL CHROMOSOME COUNT (Expected 46, Got {total_chroms}). POTENTIAL SAMPLE LOSS."
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(final_karyogram, warning_text, (50, 100), font, 1.5, (0, 0, 255), 4, cv2.LINE_AA)
            
            # Ghi ảnh có đường dẫn Unicode (Tiếng Việt)
            is_success, im_buf_arr = cv2.imencode('.png', final_karyogram)
            if is_success:
                im_buf_arr.tofile(str(output_path))
            print(f"\n⚠️ {warning_text}")
    # ==========================================
    
    return output_path
