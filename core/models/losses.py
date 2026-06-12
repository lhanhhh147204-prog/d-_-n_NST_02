# ============================================================
# FILE: core/models/losses.py
# CHỨC NĂNG: Các hàm Loss Function cho huấn luyện U-Net
# TÁCH TỪ: core/model_unet.py (chỉ phần loss functions)
# ============================================================

import tensorflow as tf
from tensorflow import keras
from config.settings import NUM_OUTPUT_CHANNELS, CHANNEL_WEIGHTS


# Khai báo tf.constant dựa trên CHANNEL_WEIGHTS từ config
TF_CHANNEL_WEIGHTS = tf.constant(CHANNEL_WEIGHTS, dtype=tf.float32)

# Các biến global này sẽ được assign() lại trong pipeline/step02_trainer.py khi compile
CURRENT_HARD_LOSS_WEIGHT = tf.Variable(0.65, trainable=False, dtype=tf.float32)
CURRENT_DISTILL_LOSS_WEIGHT = tf.Variable(0.35, trainable=False, dtype=tf.float32)


# =========================================================
# LOSS FUNCTIONS
# =========================================================

def weighted_bce(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true[..., :NUM_OUTPUT_CHANNELS], tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce * TF_CHANNEL_WEIGHTS)


def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor, smooth: float = 1e-6) -> tf.Tensor:
    y_true = tf.cast(y_true[..., :NUM_OUTPUT_CHANNELS], tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2])
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    weighted = dice * TF_CHANNEL_WEIGHTS
    weighted = tf.reduce_sum(weighted, axis=-1) / tf.reduce_sum(TF_CHANNEL_WEIGHTS)
    return 1.0 - tf.reduce_mean(weighted)


def hybrid_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return weighted_bce(y_true, y_pred) + dice_loss(y_true, y_pred)


# =========================================================
# DISTILLATION LOSS (TEACHER -> STUDENT)
# =========================================================

def student_distill_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Học 65% từ ground-truth, 35% từ soft-output của Teacher.
    y_true có 8 channels: [hard_A, B, C, Edge, soft_A, B, C, Edge].
    """
    hard = y_true[..., :NUM_OUTPUT_CHANNELS]
    soft = y_true[..., NUM_OUTPUT_CHANNELS:NUM_OUTPUT_CHANNELS * 2]
    hard_loss = hybrid_loss(hard, y_pred)

    mse = tf.square(tf.cast(soft, tf.float32) - tf.cast(y_pred, tf.float32))
    distill = tf.reduce_mean(mse * TF_CHANNEL_WEIGHTS)
    return CURRENT_HARD_LOSS_WEIGHT * hard_loss + CURRENT_DISTILL_LOSS_WEIGHT * distill
