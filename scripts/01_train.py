"""
01_train.py
===========
Train MobileNetV2 untuk klasifikasi uang Rupiah (7 kelas).

Pipeline:
  1. Load data dari folder hasil 00_merge_and_crop.py
  2. Augmentasi aman (flip, rotation, brightness, zoom) — TIDAK pakai hue/saturation shift
  3. Fine-tune MobileNetV2 (ImageNet) → 2 tahap:
       - Tahap 1 (frozen base): Train head saja, lr=1e-3
       - Tahap 2 (unfreeze top layers): Fine-tune 80 layer terakhir, lr=1e-5
  4. Simpan best model (.keras) + class mapping (labels.txt)
  5. Plot training history

Usage:
    python scripts/01_train.py \
        --data data/classification \
        --output models \
        --epochs 30 \
        --batch-size 32 \
        --img-size 224 \
        --unfreeze-epochs 20
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # suppress TF info logs

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# ─── Config ────────────────────────────────────────────────────────────────────

CLASS_ORDER = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
IMG_SIZE = 224  # MobileNetV2 default
AUTOTUNE = tf.data.AUTOTUNE


# ─── Data loaders ──────────────────────────────────────────────────────────────

def make_dataset(data_dir: Path, split: str, img_size: int, batch_size: int, augment: bool):
    """Build tf.data.Dataset dari folder split."""
    split_dir = data_dir / split
    class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    print(f"  [{split}] kelas ditemukan: {class_names}")

    # Pastikan urutan kelas konsisten
    assert set(class_names) == set(CLASS_ORDER), (
        f"Kelas di folder tidak cocok. Ditemukan: {class_names}\nExpected: {CLASS_ORDER}"
    )
    class_to_idx = {cls: i for i, cls in enumerate(CLASS_ORDER)}

    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0  # MobileNetV2 expects [-1, 1]
        return img, label

    def augment_image(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)  # uang bisa di-rotate
        img = tf.image.random_brightness(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        # Random zoom via crop + resize
        crop_size = tf.random.uniform([], minval=int(img_size * 0.85), maxval=img_size, dtype=tf.int32)
        img = tf.image.random_crop(img, size=[crop_size, crop_size, 3])
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.clip_by_value(img, -1.0, 1.0)
        return img, label

    # Collect all file paths
    paths, labels = [], []
    for cls_name in CLASS_ORDER:
        cls_dir = split_dir / cls_name
        if not cls_dir.exists():
            continue
        idx = class_to_idx[cls_name]
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                paths.append(str(img_path))
                labels.append(idx)

    print(f"  [{split}] total: {len(paths)} gambar")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if augment:
        ds = ds.shuffle(buffer_size=min(len(paths), 5000), seed=42)
    ds = ds.map(load_image, num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(augment_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, len(paths)


# ─── Model ─────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, img_size: int):
    """MobileNetV2 + custom classification head."""
    base = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # Frozen di tahap 1

    inputs = keras.Input(shape=(img_size, img_size, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model, base


# ─── Training ──────────────────────────────────────────────────────────────────

def compile_and_fit(
    model,
    train_ds,
    val_ds,
    epochs: int,
    lr: float,
    output_dir: Path,
    stage: str,
    class_weight: dict = None,
    callbacks_extra: list = None,
):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    ckpt_path = output_dir / f"best_{stage}.keras"
    callbacks = [
        ModelCheckpoint(
            str(ckpt_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        CSVLogger(str(output_dir / f"history_{stage}.csv")),
    ]
    if callbacks_extra:
        callbacks.extend(callbacks_extra)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def plot_history(history_stage1, history_stage2, output_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    acc1 = history_stage1.history["accuracy"]
    val_acc1 = history_stage1.history["val_accuracy"]
    acc2 = history_stage2.history["accuracy"]
    val_acc2 = history_stage2.history["val_accuracy"]

    all_acc = acc1 + acc2
    all_val_acc = val_acc1 + val_acc2
    boundary = len(acc1)

    axes[0].plot(all_acc, label="Train Acc")
    axes[0].plot(all_val_acc, label="Val Acc")
    axes[0].axvline(x=boundary, color="gray", linestyle="--", label="Fine-tune start")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[0].set_xlabel("Epoch")

    loss1 = history_stage1.history["loss"]
    val_loss1 = history_stage1.history["val_loss"]
    loss2 = history_stage2.history["loss"]
    val_loss2 = history_stage2.history["val_loss"]

    all_loss = loss1 + loss2
    all_val_loss = val_loss1 + val_loss2

    axes[1].plot(all_loss, label="Train Loss")
    axes[1].plot(all_val_loss, label="Val Loss")
    axes[1].axvline(x=boundary, color="gray", linestyle="--", label="Fine-tune start")
    axes[1].set_title("Loss")
    axes[1].legend()
    axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    out_path = output_dir / "training_history.png"
    plt.savefig(str(out_path), dpi=150)
    print(f"\n📊 Training history plot tersimpan: {out_path}")
    plt.close()


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 untuk klasifikasi Rupiah")
    parser.add_argument("--data", default="data/classification", help="Path ke folder classification dataset")
    parser.add_argument("--output", default="models", help="Output folder untuk model")
    parser.add_argument("--epochs", type=int, default=30, help="Epoch tahap 1 (frozen base)")
    parser.add_argument("--unfreeze-epochs", type=int, default=20, help="Epoch tahap 2 (fine-tune)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--unfreeze-layers", type=int, default=80, help="Jumlah layer terakhir yang di-unfreeze di tahap 2")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU/CPU info
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\n🖥️  TensorFlow {tf.__version__}")
    print(f"   GPUs: {len(gpus)} {'(GPU training aktif! 🚀)' if gpus else '(CPU-only, akan lambat)'}")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # ── Load datasets ──
    print("\n📂 Loading datasets...")
    train_ds, n_train = make_dataset(data_dir, "train", args.img_size, args.batch_size, augment=True)
    val_ds, n_val = make_dataset(data_dir, "val", args.img_size, args.batch_size, augment=False)
    test_ds, n_test = make_dataset(data_dir, "test", args.img_size, args.batch_size, augment=False)

    # Class weights untuk handle imbalance
    class_counts = {cls: 0 for cls in CLASS_ORDER}
    for cls_name in CLASS_ORDER:
        cls_dir = data_dir / "train" / cls_name
        if cls_dir.exists():
            class_counts[cls_name] = len(list(cls_dir.iterdir()))

    total = sum(class_counts.values())
    class_weight = {i: total / (len(CLASS_ORDER) * count) if count > 0 else 1.0
                    for i, (cls, count) in enumerate(class_counts.items())}
    print(f"\n⚖️  Class weights: {class_weight}")

    # Save class info
    class_info = {
        "classes": CLASS_ORDER,
        "class_to_idx": {cls: i for i, cls in enumerate(CLASS_ORDER)},
        "idx_to_class": {str(i): cls for i, cls in enumerate(CLASS_ORDER)},
        "img_size": args.img_size,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": n_test,
    }
    with open(output_dir / "class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)

    # ── Build model ──
    print(f"\n🏗️  Membangun model MobileNetV2...")
    model, base_model = build_model(len(CLASS_ORDER), args.img_size)
    model.summary(line_length=90)

    # ══════════════════════════════════════════════════════════════════
    # TAHAP 1: Train head saja (base frozen)
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  TAHAP 1: Training head (base model frozen)")
    print(f"  Epochs: {args.epochs}, LR: 1e-3")
    print(f"{'='*60}")

    history1 = compile_and_fit(
        model, train_ds, val_ds,
        epochs=args.epochs,
        lr=1e-3,
        output_dir=output_dir,
        stage="stage1",
        class_weight=class_weight,
    )

    # ══════════════════════════════════════════════════════════════════
    # TAHAP 2: Fine-tune top N layers
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  TAHAP 2: Fine-tuning {args.unfreeze_layers} layer terakhir")
    print(f"  Epochs: {args.unfreeze_epochs}, LR: 1e-5")
    print(f"{'='*60}")

    # Unfreeze top N layers dari base model
    base_model.trainable = True
    for layer in base_model.layers[: -args.unfreeze_layers]:
        layer.trainable = False

    trainable_count = sum(1 for l in base_model.layers if l.trainable)
    print(f"  Trainable layers di base: {trainable_count}/{len(base_model.layers)}")

    history2 = compile_and_fit(
        model, train_ds, val_ds,
        epochs=args.unfreeze_epochs,
        lr=1e-5,
        output_dir=output_dir,
        stage="stage2",
        class_weight=class_weight,
    )

    # ── Evaluation ──
    print("\n📊 Evaluasi pada test set:")
    best_model_path = output_dir / "best_stage2.keras"
    if not best_model_path.exists():
        best_model_path = output_dir / "best_stage1.keras"
    final_model = keras.models.load_model(str(best_model_path))
    test_loss, test_acc = final_model.evaluate(test_ds, verbose=1)
    print(f"\n✅ Test accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"   Test loss:     {test_loss:.4f}")

    # Save final model as SavedModel
    final_path = output_dir / "rupiah_mobilenetv2"
    final_model.save(str(final_path))
    print(f"\n💾 Model tersimpan: {final_path}")

    # Plot
    plot_history(history1, history2, output_dir)

    # Save hasil evaluasi
    results = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "classes": CLASS_ORDER,
        "img_size": args.img_size,
        "best_model": str(best_model_path),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n🎉 Training selesai! Model terbaik: {best_model_path}")
    print(f"   Jalankan 02_export_tflite.py untuk ekspor ke TFLite.")


if __name__ == "__main__":
    main()
