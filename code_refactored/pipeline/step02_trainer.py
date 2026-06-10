# ============================================================
# FILE: pipeline/02_trainer.py
# CHỨC NĂNG: Xây dựng Data Loader và Huấn Luyện U-Net
# ============================================================

import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from config.settings import (
    DATASET_DIR, RESULTS_DIR, IMG_SIZE, NUM_OUTPUT_CHANNELS, 
    TEACHER_BASE_FILTERS, STUDENT_BASE_FILTERS, ABC_THRESHOLDS,
    DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_PATIENCE
)
from core.model_unet import (
    build_unet, hybrid_loss, student_distill_loss, 
    CURRENT_HARD_LOSS_WEIGHT, CURRENT_DISTILL_LOSS_WEIGHT
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
AUTOTUNE = tf.data.AUTOTUNE

def get_file_lists(dataset_dir: Path, split: str):
    split_dir = dataset_dir / split
    img_dir = split_dir / "images"
    imgs = sorted(img_dir.glob("*.png"))
    
    paths_img, paths_a, paths_b, paths_c = [], [], [], []
    for img_path in imgs:
        name = img_path.name
        ma = split_dir / "masks_A" / name
        mb = split_dir / "masks_B" / name
        mc = split_dir / "masks_C" / name
        if ma.exists() and mb.exists() and mc.exists():
            paths_img.append(str(img_path))
            paths_a.append(str(ma))
            paths_b.append(str(mb))
            paths_c.append(str(mc))
            
    return paths_img, paths_a, paths_b, paths_c

def read_image(path: tf.Tensor) -> tf.Tensor:
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img

def read_mask(path: tf.Tensor) -> tf.Tensor:
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
    mask = tf.cast(mask > 127, tf.float32)
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return mask

def make_boundary_channel(mask_abc: tf.Tensor, k: int = 3) -> tf.Tensor:
    fg = tf.reduce_max(mask_abc, axis=-1, keepdims=True)
    x = tf.expand_dims(fg, axis=0)
    dil = tf.nn.max_pool2d(x, ksize=k, strides=1, padding="SAME")
    ero = 1.0 - tf.nn.max_pool2d(1.0 - x, ksize=k, strides=1, padding="SAME")
    edge = tf.clip_by_value(dil - ero, 0.0, 1.0)
    return tf.squeeze(edge, axis=0)

def load_sample(image_path, ma_path, mb_path, mc_path):
    img = read_image(image_path)
    ma, mb, mc = read_mask(ma_path), read_mask(mb_path), read_mask(mc_path)
    mask_abc = tf.concat([ma, mb, mc], axis=-1)
    edge = make_boundary_channel(mask_abc)
    mask_abce = tf.concat([mask_abc, edge], axis=-1)
    mask_abce.set_shape([IMG_SIZE, IMG_SIZE, NUM_OUTPUT_CHANNELS])
    return img, mask_abce

def make_dataset(split: str, batch_size: int, shuffle: bool = False):
    p_img, p_a, p_b, p_c = get_file_lists(DATASET_DIR, split)
    n = len(p_img)
    ds = tf.data.Dataset.from_tensor_slices((p_img, p_a, p_b, p_c))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(n, 4096))
    ds = ds.map(load_sample, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(1)
    return ds, n

def make_distill_dataset(ds: tf.data.Dataset, teacher: keras.Model) -> tf.data.Dataset:
    teacher.trainable = False
    def add_soft(x, y_hard):
        y_soft = tf.stop_gradient(teacher(x, training=False))
        y = tf.concat([y_hard, y_soft], axis=-1)
        y.set_shape([None, IMG_SIZE, IMG_SIZE, NUM_OUTPUT_CHANNELS * 2])
        return x, y
    return ds.map(add_soft, num_parallel_calls=1).prefetch(1)

def train_model(role="teacher", epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_ds, _ = make_dataset("train", batch_size, shuffle=True)
    val_ds, _ = make_dataset("val", batch_size, shuffle=False)
    
    if role == "teacher":
        model = build_unet(base_filters=TEACHER_BASE_FILTERS, name="Teacher_UNet")
        model.compile(optimizer=keras.optimizers.Adam(DEFAULT_LEARNING_RATE), loss=hybrid_loss)
        ds = train_ds
    else:
        teacher_path = RESULTS_DIR / "best_teacher.keras"
        if not teacher_path.exists():
            print("❌ Chưa có Teacher model. Vui lòng train teacher trước.")
            return
        teacher = keras.models.load_model(teacher_path, compile=False)
        model = build_unet(base_filters=STUDENT_BASE_FILTERS, name="Student_UNet")
        model.compile(optimizer=keras.optimizers.Adam(DEFAULT_LEARNING_RATE), loss=student_distill_loss)
        ds = make_distill_dataset(train_ds, teacher)
        val_ds = make_distill_dataset(val_ds, teacher)
        
    callbacks = [
        keras.callbacks.ModelCheckpoint(str(RESULTS_DIR / f"best_{role}.keras"), monitor="val_loss", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=DEFAULT_PATIENCE, restore_best_weights=True)
    ]
    
    print(f"🚀 Bắt đầu huấn luyện {role.upper()}...")
    model.fit(ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    print(f"✅ Hoàn thành huấn luyện {role.upper()}.")

if __name__ == "__main__":
    import sys
    role = "teacher"
    if len(sys.argv) > 1:
        role = sys.argv[1]
    train_model(role)
