# ============================================================
# FILE: core/__init__.py
# CHỨC NĂNG: Export tổng hợp toàn bộ lớp core
# ============================================================

# Kiến trúc mạng & Loss Functions
from medical_pipeline.core.models.unet import build_unet
from medical_pipeline.core.models.losses import (
    hybrid_loss,
    weighted_bce,
    dice_loss,
    student_distill_loss,
    TF_CHANNEL_WEIGHTS,
    CURRENT_HARD_LOSS_WEIGHT,
    CURRENT_DISTILL_LOSS_WEIGHT,
)

# Metrics đánh giá
from medical_pipeline.core.metrics.segmentation import (
    calculate_dice,
    calculate_iou,
    calculate_pixel_accuracy,
)
