# ============================================================
# FILE: main.py
# CHỨC NĂNG: Giao diện dòng lệnh CLI cho Toàn Bộ Dự Án NST
# ============================================================

import argparse
from pathlib import Path

from pipeline.step01_data_prep import run_all as run_data_prep
from pipeline.step02_trainer import train_model
from pipeline.step03_inference import run_inference
from pipeline.step04_evaluator import evaluate_test_set

def main():
    parser = argparse.ArgumentParser(description="Chương trình Phân Tích Hình Thái Nhiễm Sắc Thể (NST)")
    parser.add_argument(
        "--step", 
        type=str, 
        required=True,
        choices=["prepare_data", "train_teacher", "train_student", "evaluate", "predict", "all"],
        help="Chọn bước để chạy trong Pipeline."
    )
    
    # Tham số cho Train
    parser.add_argument("--epochs", type=int, default=200, help="Số epochs huấn luyện (mặc định: 200).")
    parser.add_argument("--batch-size", type=int, default=50, help="Kích thước batch (mặc định: 50).")
    
    # Tham số cho Predict
    parser.add_argument("--input-dir", type=str, default=None, help="Thư mục chứa ảnh NST thực tế (cho bước predict).")
    parser.add_argument("--model-path", type=str, default=None, help="Đường dẫn file .keras cụ thể để predict/evaluate.")

    args = parser.parse_args()

    print("==================================================")
    print("🧬 DỰ ÁN PHÂN TÍCH NST (Trạng thái: Tách, Trùng, Uốn)")
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
        # Mặc định lấy từ setting, nếu truyền vào thì lấy theo args
        from config.settings import RAW_OVERLAP_DIR
        input_d = Path(args.input_dir) if args.input_dir else RAW_OVERLAP_DIR
        model_p = Path(args.model_path) if args.model_path else None
        run_inference(input_dir=input_d, model_path=model_p)

    print("\n✅ TẤT CẢ CÁC BƯỚC YÊU CẦU ĐÃ HOÀN TẤT!")

if __name__ == "__main__":
    main()
