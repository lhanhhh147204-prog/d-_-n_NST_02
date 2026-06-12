# ============================================================
# FILE: data/dataset.py
# CHỨC NĂNG: TF Data Pipeline (đọc, ghép, augment, tạo dataset)
# TÁCH TỪ: pipeline/step02_trainer.py (phần data loading)
# ============================================================

from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from config.settings import DATASET_DIR, IMG_SIZE, NUM_OUTPUT_CHANNELS


AUTOTUNE = tf.data.AUTOTUNE


def get_file_lists(dataset_dir: Path, split: str):
    """Lấy danh sách đường dẫn file ảnh và mask theo split (train/val/test)."""
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
    """Đọc ảnh PNG xám từ path, chuẩn hóa về [0,1] và resize về IMG_SIZE."""
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img


def read_mask(path: tf.Tensor) -> tf.Tensor:
    """Đọc mask PNG từ path, nhị phân hóa (>127 = 1) và resize về IMG_SIZE."""
    mask = tf.io.read_file(path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, (IMG_SIZE, IMG_SIZE), method="nearest")
    mask = tf.cast(mask > 127, tf.float32)
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return mask


def make_boundary_channel(mask_abc: tf.Tensor, k: int = 3) -> tf.Tensor:
    """Tạo edge channel từ mask A+B+C bằng cách dilation - erosion."""
    fg = tf.reduce_max(mask_abc, axis=-1, keepdims=True)
    x = tf.expand_dims(fg, axis=0)
    dil = tf.nn.max_pool2d(x, ksize=k, strides=1, padding="SAME")
    ero = 1.0 - tf.nn.max_pool2d(1.0 - x, ksize=k, strides=1, padding="SAME")
    edge = tf.clip_by_value(dil - ero, 0.0, 1.0)
    return tf.squeeze(edge, axis=0)


def load_sample(image_path, ma_path, mb_path, mc_path):
    """Đọc và ghép ảnh + mask A/B/C/Edge thành một sample hoàn chỉnh."""
    img = read_image(image_path)
    ma, mb, mc = read_mask(ma_path), read_mask(mb_path), read_mask(mc_path)
    mask_abc = tf.concat([ma, mb, mc], axis=-1)
    edge = make_boundary_channel(mask_abc)
    mask_abce = tf.concat([mask_abc, edge], axis=-1)
    mask_abce.set_shape([IMG_SIZE, IMG_SIZE, NUM_OUTPUT_CHANNELS])
    return img, mask_abce


def make_dataset(split: str, batch_size: int, shuffle: bool = False, dataset_dir: Path = None):
    """Tạo tf.data.Dataset cho một split nhất định."""
    if dataset_dir is None:
        dataset_dir = DATASET_DIR
    p_img, p_a, p_b, p_c = get_file_lists(dataset_dir, split)
    n = len(p_img)
    ds = tf.data.Dataset.from_tensor_slices((p_img, p_a, p_b, p_c))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(n, 4096))
    ds = ds.map(load_sample, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(1)
    return ds, n


def make_distill_dataset(ds: tf.data.Dataset, teacher: keras.Model) -> tf.data.Dataset:
    """Wrap dataset với soft-label sinh từ Teacher model (dùng cho Knowledge Distillation)."""
    teacher.trainable = False
    def add_soft(x, y_hard):
        y_soft = tf.stop_gradient(teacher(x, training=False))
        y = tf.concat([y_hard, y_soft], axis=-1)
        y.set_shape([None, IMG_SIZE, IMG_SIZE, NUM_OUTPUT_CHANNELS * 2])
        return x, y
    return ds.map(add_soft, num_parallel_calls=1).prefetch(1)
