# ============================================================
# FILE: config/settings.py
# CHỨC NĂNG: Nơi lưu trữ TOÀN BỘ cấu hình (siêu tham số, đường dẫn)
# ============================================================

from pathlib import Path

# ==========================================
# 1. PATHS (ĐƯỜNG DẪN THƯ MỤC)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input thô
SOURCE_DIR = PROJECT_ROOT / "data_storage" / "source_data"
RAW_OVERLAP_DIR = SOURCE_DIR / "overlap_raw"
RAW_SINGLE_DIR = SOURCE_DIR / "single_chromosomes"

# Dữ liệu trung gian
PREP_SINGLE_DIR = PROJECT_ROOT / "prepared_single_chromosomes"
GEN_DATA_DIR = PROJECT_ROOT / "generated_data"
PROCESSED_256_DIR = PROJECT_ROOT / "processed_data_256"

# Dataset chính thức (Train/Val/Test)
DATASET_DIR = PROJECT_ROOT / "data_storage" / "dataset"

# Kết quả và Model
RESULTS_DIR = PROJECT_ROOT / "results"

# ==========================================
# 2. IMAGE PROCESSING (XỬ LÝ ẢNH)
# ==========================================
IMG_SIZE = 256  # Tất cả ảnh đưa vào model phải là 256x256

# Tham số tách NST đơn (01_data_prep.py - part 1)
BG_MARGIN = 15
CROP_PADDING = 5
KERNEL_SIZE = 5
MASK_SHRINK_ITER = 1

# Tham số sinh dữ liệu tổng hợp (01_data_prep.py - part 2)
CANVAS_SIZE = 400
TARGET_LONG_SIDE_MIN = 120
TARGET_LONG_SIDE_MAX = 200
MIN_OVERLAP_PIXELS = 120
MAX_OVERLAP_RATIO = 0.55
TOTAL_SAMPLES_TO_GENERATE = 3000

# Tham số chia Train/Val/Test (01_data_prep.py - part 3)
SPLIT_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# Test_ratio = 1.0 - train - val

# ==========================================
# 3. MODEL ARCHITECTURE (KIẾN TRÚC MÔ HÌNH)
# ==========================================
NUM_OUTPUT_CHANNELS = 4  # [A, B, C, Edge]

# Teacher-Student setup
TEACHER_BASE_FILTERS = 48
STUDENT_BASE_FILTERS = 24

# Channel Weights (C cực kỳ quan trọng)
# [A, B, C, Edge]
CHANNEL_WEIGHTS = [1.25, 1.25, 3.75, 1.00]

# Distillation Weights
HARD_LOSS_WEIGHT_DEFAULT = 0.65
DISTILL_LOSS_WEIGHT_DEFAULT = 0.35

# ==========================================
# 4. INFERENCE & EVALUATION (DỰ ĐOÁN & ĐÁNH GIÁ)
# ==========================================
# Thresholds cho 3 channel chính: A, B, C
ABC_THRESHOLDS = [0.50, 0.50, 0.40]

# ==========================================
# 5. TRAINING (CẤU HÌNH TRAIN CƠ BẢN)
# ==========================================
DEFAULT_EPOCHS = 200
DEFAULT_BATCH_SIZE = 50
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_PATIENCE = 30

# ==========================================
# 6. PYTORCH WEIGHTS (ĐƯỜNG DẪN MÔ HÌNH PYTORCH)
# ==========================================
WEIGHTS_DIR = PROJECT_ROOT / "ai_training" / "weights"

# Weights cho module khử mờ MPRNet
DEBLURRER_WEIGHTS = WEIGHTS_DIR / "deblurring_best.pth"

# Weights cho module phân loại CCINet
CCINET_WEIGHTS = WEIGHTS_DIR / "ccinet_best.pth"

# Weights cho module phân loại 24 lớp NST (Swin-T)
KARYOTYPE_CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "karyotype_swint_best.pth"

# ==========================================
# 7. KARYOTYPE CLASSIFIER (BỘ PHÂN LOẠI 24 LỚP NST)
# ==========================================
NUM_KARYOTYPE_CLASSES = 24  # 22 NST thường + X + Y

# Nhãn chuẩn y khoa cho 24 lớp NST
KARYOTYPE_LABELS = [str(i) for i in range(1, 23)] + ["X", "Y"]

# Cấu hình huấn luyện classifier
KARYOTYPE_TRAIN_EPOCHS = 150  # Tăng lên để đủ thời gian cho 2-phase training
KARYOTYPE_TRAIN_BATCH_SIZE = 32
KARYOTYPE_TRAIN_LR = 3e-4
KARYOTYPE_IMG_SIZE = 224  # Kích thước chuẩn cho Swin-T

# Thư mục dataset phân loại (mỗi lớp = 1 thư mục con)
KARYOTYPE_DATASET_DIR = DATASET_DIR / "karyotype"

# ==========================================
# 8. KARYOGRAM (VẼ BẢNG XẾP NST CHUẨN Y KHOA)
# ==========================================
KARYOGRAM_OUTPUT_DIR = RESULTS_DIR / "karyograms"

# Nhóm NST hiển thị Karyogram tối giản (Chuẩn Clinical 4 hàng)
DENVER_GROUPS = {
    "Row 1": ["1", "2", "3", "4", "5"],
    "Row 2": ["6", "7", "8", "9", "10", "11", "12"],
    "Row 3": ["13", "14", "15", "16", "17", "18"],
    "Row 4": ["19", "20", "21", "22", "X", "Y"],
}
