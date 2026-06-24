# ============================================================
# FILE: main.py
# CHỨC NĂNG: Giao diện dòng lệnh CLI cho Toàn Bộ Du An NST
# ============================================================

import sys
import io
import argparse
from pathlib import Path

# Fix encoding cho Windows terminal (CP1252 -> UTF-8)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ('utf-8', 'utf8'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from ai_training.training.data_prep import run_all as run_data_prep
from ai_training.training.trainer import train_model
from ai_training.training.evaluator import evaluate_test_set

def main():
    parser = argparse.ArgumentParser(
        description="Dự Án Phân Tích Hình Thái Nhiễm Sắc Thể (NST) - Y Khoa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["prepare_data", "train_teacher", "train_student", "evaluate", "predict",
                "karyotype", "karyogram", "full_pipeline", "all"],
        help=("Chọn bước chạy: prepare_data | train_teacher | train_student "
              "| evaluate | predict | karyotype | karyogram | full_pipeline | all")
    )
    
    # Tham so cho Train
    parser.add_argument("--epochs", type=int, default=200, help="Số epochs huấn luyện (mặc định: 200).")
    parser.add_argument("--batch-size", type=int, default=50, help="Kích thước batch (mặc định: 50).")

    # Tham so cho Predict / Evaluate
    parser.add_argument("--input-dir", type=str, default=None, help="Thư mục ảnh (hoặc đường dẫn file gốc).")
    parser.add_argument("--model-path", type=str, default=None, help="Đường dẫn file .keras/.pth để predict/evaluate.")

    # Tham so cho Karyotype
    parser.add_argument("--sex", type=str, default="XX", choices=["XX", "XY"],
                        help="Giới tính mẫu (XX=nữ, XY=nam). Mặc định: XX.")

    args = parser.parse_args()

    print("==================================================")
    print("[NST] DỰ ÁN PHÂN TÍCH NHIỄM SẮC THỂ (PIPELINE Y KHOA)")
    print("==================================================\n")

    if args.step in ["prepare_data", "all"]:
        run_data_prep()

    if args.step in ["train_teacher", "all"]:
        train_model(role="teacher", epochs=args.epochs, batch_size=args.batch_size)

    if args.step in ["train_student", "all"]:
        train_model(role="student", epochs=args.epochs, batch_size=args.batch_size)

    if args.step in ["evaluate", "all"]:
        model_p = Path(args.model_path) if args.model_path else None
        evaluate_test_set(model_path=model_p)

    if args.step in ["predict", "all"]:
        # Tách cụm chồng lấp
        from config.settings import RAW_OVERLAP_DIR
        input_d = Path(args.input_dir) if args.input_dir else RAW_OVERLAP_DIR
        from config.settings import RESULTS_DIR
        out_d = RESULTS_DIR / "real_predictions" / "separated_chromosomes"
        from medical_pipeline.pipeline.buoc3_main import run_overlap_separation
        run_overlap_separation(input_dir=input_d, output_dir=out_d)

    if args.step in ["karyotype", "all"]:
        # Chạy pipeline karyotyping từ ảnh NST đơn (Bước 4,5,7,8)
        from config.settings import RESULTS_DIR
        input_d = Path(args.input_dir) if args.input_dir else RESULTS_DIR / "real_predictions" / "separated_chromosomes"
        from medical_pipeline.pipeline.buoc9_tong_hop import _load_separated_images
        from medical_pipeline.pipeline.buoc4_duoi_thang_nhiem_sac_the import run_straightening
        from medical_pipeline.pipeline.buoc5_phan_loai_nhiem_sac_theo_loai import run_classification_by_type
        from medical_pipeline.pipeline.buoc7_ghep_cap_hoan_hao import run_perfect_pairing
        from medical_pipeline.pipeline.buoc8_ve_karyogram import run_render_karyogram
        
        chromosomes = _load_separated_images(input_d)
        if chromosomes:
            chromosomes = run_straightening(chromosomes)
            chromosomes = run_classification_by_type(chromosomes)
            pairing = run_perfect_pairing(chromosomes, sex=args.sex)
            run_render_karyogram(chromosomes, pairing, sex=args.sex)

    if args.step == "karyogram":
        # Chỉ vẽ Karyogram từ ảnh đã phân loại (bỏ qua classify + pair)
        from karyogram.builder import build_karyogram_from_files
        from config.settings import KARYOGRAM_OUTPUT_DIR
        input_d = Path(args.input_dir) if args.input_dir else KARYOGRAM_OUTPUT_DIR
        output_p = KARYOGRAM_OUTPUT_DIR / f"karyogram_{args.sex}.png"
        build_karyogram_from_files(input_d, sex=args.sex, output_path=output_p)

    if args.step in ["full_pipeline", "all"]:
        # Chạy từ ảnh gốc tới Karyogram
        from medical_pipeline.pipeline.buoc9_tong_hop import run_end_to_end_up_to_step7
        if not args.input_dir:
            print("❌ Vui lòng cung cấp --input-dir trỏ tới 1 file ảnh tế bào nguyên vẹn để chạy full_pipeline.")
        else:
            run_end_to_end_up_to_step7(args.input_dir, sex=args.sex)

    print("\n[OK] TẤT CẢ CÁC BƯỚC ĐÃ HOÀN TẤT!")

if __name__ == "__main__":
    main()
