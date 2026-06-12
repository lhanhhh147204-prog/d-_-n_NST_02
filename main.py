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

from pipeline.step01_data_prep import run_all as run_data_prep
from pipeline.step02_trainer import train_model
from pipeline.step03_inference import run_inference
from pipeline.step04_evaluator import evaluate_test_set

def main():
    parser = argparse.ArgumentParser(
        description="Du An Phan Tich Hinh Thai Nhiem Sac The (NST)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--step",
        type=str,
        required=True,
        choices=["prepare_data", "train_teacher", "train_student", "evaluate", "predict", "all"],
        help=("Chon buoc chay: prepare_data | train_teacher | train_student "
              "| evaluate | predict | all")
    )
    
    # Tham so cho Train
    parser.add_argument("--epochs", type=int, default=200, help="So epochs huan luyen (mac dinh: 200).")
    parser.add_argument("--batch-size", type=int, default=50, help="Kich thuoc batch (mac dinh: 50).")

    # Tham so cho Predict / Evaluate
    parser.add_argument("--input-dir", type=str, default=None, help="Thu muc anh NST thuc te (cho buoc predict).")
    parser.add_argument("--model-path", type=str, default=None, help="Duong dan file .keras de predict/evaluate.")

    args = parser.parse_args()

    print("==================================================")
    print("[NST] DU AN PHAN TICH NHIEM SAC THE")
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

    print("\n[OK] TAT CA CAC BUOC YEU CAU DA HOAN TAT!")

if __name__ == "__main__":
    main()
