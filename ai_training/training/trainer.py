# ============================================================
# FILE: pipeline/step02_trainer.py
# CHỨC NĂNG: Controller — Xây dựng và Huấn Luyện U-Net
# Logic data pipeline nằm trong: data/dataset.py
# Logic model/loss nằm trong: core/models/
# ============================================================

import os
from tensorflow import keras

from config.settings import (
    RESULTS_DIR, DEFAULT_EPOCHS, DEFAULT_BATCH_SIZE, DEFAULT_LEARNING_RATE, DEFAULT_PATIENCE,
    TEACHER_BASE_FILTERS, STUDENT_BASE_FILTERS,
)
from medical_pipeline.core.models.unet import build_unet
from medical_pipeline.core.models.losses import hybrid_loss, student_distill_loss
from medical_pipeline.data_modules.dataset import make_dataset, make_distill_dataset

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")


def train_model(role: str = "teacher", epochs: int = DEFAULT_EPOCHS, batch_size: int = DEFAULT_BATCH_SIZE):
    """
    Huấn luyện Teacher hoặc Student U-Net.
    - role='teacher': huấn luyện với hybrid_loss trên ground-truth.
    - role='student': huấn luyện với student_distill_loss (cần Teacher đã train).
    """
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
        keras.callbacks.ModelCheckpoint(
            str(RESULTS_DIR / f"best_{role}.keras"),
            monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=DEFAULT_PATIENCE, restore_best_weights=True
        ),
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
