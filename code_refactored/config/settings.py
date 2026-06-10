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
SOURCE_DIR = PROJECT_ROOT / "source_data"
RAW_OVERLAP_DIR = SOURCE_DIR / "overlap_raw"
RAW_SINGLE_DIR = SOURCE_DIR / "single_chromosomes"

# Dữ liệu trung gian
PREP_SINGLE_DIR = PROJECT_ROOT / "prepared_single_chromosomes"
GEN_DATA_DIR = PROJECT_ROOT / "generated_data"
PROCESSED_256_DIR = PROJECT_ROOT / "processed_data_256"

# Dataset chính thức (Train/Val/Test)
DATASET_DIR = PROJECT_ROOT / "dataset"

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
