# ============================================================
# FILE: pipeline/04_evaluator.py
# CHỨC NĂNG: Đánh giá mô hình trên tập Test tổng hợp
# ============================================================

import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config.settings import DATASET_DIR, RESULTS_DIR, IMG_SIZE, ABC_THRESHOLDS, DEFAULT_BATCH_SIZE
from core.metrics import calculate_dice, calculate_iou, calculate_pixel_accuracy
from core.model_unet import hybrid_loss, student_distill_loss

def get_test_files():
    test_dir = DATASET_DIR / "test"
    imgs = sorted((test_dir / "images").glob("*.png"))
    paths_img, paths_a, paths_b, paths_c = [], [], [], []
    for p in imgs:
        name = p.name
        ma = test_dir / "masks_A" / name
        mb = test_dir / "masks_B" / name
        mc = test_dir / "masks_C" / name
        if ma.exists() and mb.exists() and mc.exists():
            paths_img.append(p)
            paths_a.append(ma)
            paths_b.append(mb)
            paths_c.append(mc)
    return paths_img, paths_a, paths_b, paths_c

def evaluate_test_set(model_path=None):
    if model_path is None:
        model_path = RESULTS_DIR / "best_student.keras"
        if not model_path.exists():
            model_path = RESULTS_DIR / "best_teacher.keras"
            
    if not model_path.exists():
        print(f"❌ Không tìm thấy model tại {model_path}")
        return

    print(f"Loading model: {model_path}")
    custom_objects = {
        "hybrid_loss": hybrid_loss, 
        "student_distill_loss": student_distill_loss
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

    p_img, p_a, p_b, p_c = get_test_files()
    if not p_img:
        print("⚠️ Không có ảnh test.")
        return

    print(f"Bắt đầu đánh giá {len(p_img)} ảnh test...")
    
    all_dice_A, all_dice_B, all_dice_C = [], [], []
    all_iou_A, all_iou_B, all_iou_C = [], [], []
    all_acc = []

    for i in range(0, len(p_img), DEFAULT_BATCH_SIZE):
        batch_imgs, batch_a, batch_b, batch_c = [], [], [], []
        
        for j in range(i, min(i + DEFAULT_BATCH_SIZE, len(p_img))):
            img = tf.image.decode_png(tf.io.read_file(str(p_img[j])), channels=1)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
            batch_imgs.append(img.numpy())
            
            ma = tf.image.resize(tf.image.decode_png(tf.io.read_file(str(p_a[j])), channels=1), (IMG_SIZE, IMG_SIZE), method="nearest").numpy() > 127
            mb = tf.image.resize(tf.image.decode_png(tf.io.read_file(str(p_b[j])), channels=1), (IMG_SIZE, IMG_SIZE), method="nearest").numpy() > 127
            mc = tf.image.resize(tf.image.decode_png(tf.io.read_file(str(p_c[j])), channels=1), (IMG_SIZE, IMG_SIZE), method="nearest").numpy() > 127
            
            batch_a.append(ma)
            batch_b.append(mb)
            batch_c.append(mc)

        x = np.stack(batch_imgs, axis=0)
        y_a = np.stack(batch_a, axis=0)
        y_b = np.stack(batch_b, axis=0)
        y_c = np.stack(batch_c, axis=0)
        
        preds = model.predict(x, verbose=0)
        pred_a = preds[..., 0:1] > ABC_THRESHOLDS[0]
        pred_b = preds[..., 1:2] > ABC_THRESHOLDS[1]
        pred_c = preds[..., 2:3] > ABC_THRESHOLDS[2]

        for k in range(len(batch_imgs)):
            all_dice_A.append(calculate_dice(y_a[k], pred_a[k]))
            all_dice_B.append(calculate_dice(y_b[k], pred_b[k]))
            all_dice_C.append(calculate_dice(y_c[k], pred_c[k]))
            
            all_iou_A.append(calculate_iou(y_a[k], pred_a[k]))
            all_iou_B.append(calculate_iou(y_b[k], pred_b[k]))
            all_iou_C.append(calculate_iou(y_c[k], pred_c[k]))
            
            y_abc = np.concatenate([y_a[k], y_b[k], y_c[k]], axis=-1)
            p_abc = np.concatenate([pred_a[k], pred_b[k], pred_c[k]], axis=-1)
            all_acc.append(calculate_pixel_accuracy(y_abc, p_abc))

    results = {
        "mean_dice_A": float(np.mean(all_dice_A)),
        "mean_dice_B": float(np.mean(all_dice_B)),
        "mean_dice_C": float(np.mean(all_dice_C)),
        "mean_dice_ABC": float(np.mean(all_dice_A + all_dice_B + all_dice_C)),
        "mean_iou_A": float(np.mean(all_iou_A)),
        "mean_iou_B": float(np.mean(all_iou_B)),
        "mean_iou_C": float(np.mean(all_iou_C)),
        "mean_iou_ABC": float(np.mean(all_iou_A + all_iou_B + all_iou_C)),
        "pixel_accuracy": float(np.mean(all_acc)),
    }

    print("\n📊 KẾT QUẢ ĐÁNH GIÁ (TEST SET):")
    print(json.dumps(results, indent=2))
    
    report_path = RESULTS_DIR / "test_evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Đã lưu báo cáo tại: {report_path}")

if __name__ == "__main__":
    evaluate_test_set()
