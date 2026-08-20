"""
02_export_tflite.py
===================
Ekspor Keras/SavedModel → TFLite dalam 3 varian:
  1. FP32  → rupiah_classifier.tflite        (full precision)
  2. FP16  → rupiah_classifier_fp16.tflite   (smaller, same GPU-compatible)
  3. INT8  → rupiah_classifier_int8.tflite   (paling kecil, untuk mobile)

INT8 butuh representative dataset untuk kalibrasi quantization.

Usage:
    python scripts/02_export_tflite.py \
        --model models/rupiah_mobilenetv2 \
        --data data/classification \
        --output models/tflite \
        --img-size 224 \
        --rep-samples 200
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf


CLASS_ORDER = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ─── Representative dataset ────────────────────────────────────────────────────

def make_representative_dataset(data_dir: Path, img_size: int, n_samples: int):
    """
    Generator untuk INT8 quantization calibration.
    Ambil sampel acak dari training set.
    """
    all_paths = []
    for cls in CLASS_ORDER:
        cls_dir = data_dir / "train" / cls
        if cls_dir.exists():
            paths = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
            all_paths.extend(paths)

    # Acak dan ambil n_samples
    np.random.shuffle(all_paths)
    selected = all_paths[:n_samples]

    def generator():
        for img_path in selected:
            img = tf.io.read_file(str(img_path))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1.0
            img = tf.expand_dims(img, axis=0)  # [1, H, W, 3]
            yield [img]

    return generator


# ─── Export helpers ────────────────────────────────────────────────────────────

def get_converter(model, sm_dir="/tmp/temp_saved_model"):
    try:
        model.export(sm_dir)
        return tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    except Exception:
        return tf.lite.TFLiteConverter.from_keras_model(model)


def export_fp32(model, output_path: Path):
    print("\n📦 [1/3] Ekspor FP32...")
    converter = get_converter(model, "/tmp/sm_fp32")
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"   ✅ Tersimpan: {output_path} ({size_mb:.2f} MB)")
    return size_mb


def export_fp16(model, output_path: Path):
    print("\n📦 [2/3] Ekspor FP16...")
    converter = get_converter(model, "/tmp/sm_fp16")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"   ✅ Tersimpan: {output_path} ({size_mb:.2f} MB)")
    return size_mb


def export_int8(model, output_path: Path, rep_dataset_gen, img_size: int):
    print("\n📦 [3/3] Ekspor INT8 (full integer quantization)...")
    converter = get_converter(model, "/tmp/sm_int8")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = rep_dataset_gen

    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"   ✅ Tersimpan: {output_path} ({size_mb:.2f} MB)")
    return size_mb


# ─── Benchmark / verify ────────────────────────────────────────────────────────

def verify_tflite(tflite_path: Path, data_dir: Path, img_size: int, n_samples: int = 50):
    """Run inference pada beberapa gambar, hitung top-1 accuracy."""
    print(f"\n🔍 Verifikasi {tflite_path.name}...")

    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    correct = 0
    total = 0

    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        cls_dir = data_dir / "val" / cls_name
        if not cls_dir.exists():
            continue
        imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS][:max(1, n_samples // 7)]

        for img_path in imgs:
            img = tf.io.read_file(str(img_path))
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [img_size, img_size])
            img = tf.cast(img, tf.float32) / 127.5 - 1.0
            img = np.expand_dims(img.numpy(), axis=0).astype(np.float32)

            interpreter.set_tensor(input_details[0]["index"], img)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]["index"])[0]

            pred_idx = np.argmax(output)
            if pred_idx == cls_idx:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"   Accuracy pada val subset ({total} gambar): {acc:.4f} ({acc*100:.2f}%)")
    return acc


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Ekspor model Keras → TFLite (FP32, FP16, INT8)")
    parser.add_argument("--model", default="models/rupiah_mobilenetv2", help="Path ke SavedModel atau .keras")
    parser.add_argument("--data", default="data/classification", help="Path ke folder classification dataset")
    parser.add_argument("--output", default="models/tflite", help="Output folder untuk .tflite files")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--rep-samples", type=int, default=200, help="Jumlah sampel untuk INT8 kalibrasi")
    parser.add_argument("--no-verify", action="store_true", help="Skip verifikasi akurasi setelah ekspor")
    args = parser.parse_args()

    data_dir = Path(args.data)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"🤖 TensorFlow {tf.__version__}")
    print(f"📂 Loading model: {args.model}")

    model = tf.keras.models.load_model(args.model)
    model.summary(line_length=80, print_fn=lambda x: None)  # silent summary

    input_shape = model.input_shape
    print(f"   Input shape: {input_shape}")
    print(f"   Output shape: {model.output_shape}")

    # ── Export ──
    fp32_path = output_dir / "rupiah_classifier.tflite"
    fp16_path = output_dir / "rupiah_classifier_fp16.tflite"
    int8_path = output_dir / "rupiah_classifier_int8.tflite"

    rep_gen = make_representative_dataset(data_dir, args.img_size, args.rep_samples)

    size_fp32 = export_fp32(model, fp32_path)
    size_fp16 = export_fp16(model, fp16_path)
    size_int8 = export_int8(model, int8_path, rep_gen, args.img_size)

    # ── Verify ──
    acc_fp32 = acc_fp16 = acc_int8 = None
    if not args.no_verify and (data_dir / "val").exists():
        acc_fp32 = verify_tflite(fp32_path, data_dir, args.img_size)
        acc_fp16 = verify_tflite(fp16_path, data_dir, args.img_size)
        acc_int8 = verify_tflite(int8_path, data_dir, args.img_size)

    # ── Copy labels file ──
    labels_src = data_dir / "labels.txt"
    if labels_src.exists():
        import shutil
        shutil.copy(labels_src, output_dir / "labels.txt")
        print(f"\n📄 labels.txt disalin ke {output_dir}")

    # ── Save class info ──
    class_info = {
        "classes": CLASS_ORDER,
        "idx_to_class": {str(i): cls for i, cls in enumerate(CLASS_ORDER)},
        "img_size": args.img_size,
        "input_normalization": "divide by 127.5 then subtract 1.0  →  range [-1, 1]",
    }
    with open(output_dir / "class_info.json", "w") as f:
        json.dump(class_info, f, indent=2)

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  RINGKASAN EKSPOR")
    print(f"{'='*55}")
    print(f"  {'Model':<30} {'Size':>8}  {'Val Acc':>8}")
    print(f"  {'-'*50}")
    print(f"  {'FP32  (rupiah_classifier.tflite)':<30} {size_fp32:>6.2f}MB  {f'{acc_fp32*100:.1f}%' if acc_fp32 else 'N/A':>8}")
    print(f"  {'FP16  (rupiah_classifier_fp16.tflite)':<30} {size_fp16:>6.2f}MB  {f'{acc_fp16*100:.1f}%' if acc_fp16 else 'N/A':>8}")
    print(f"  {'INT8  (rupiah_classifier_int8.tflite)':<30} {size_int8:>6.2f}MB  {f'{acc_int8*100:.1f}%' if acc_int8 else 'N/A':>8}")
    print(f"{'='*55}")
    print(f"\n💡 Rekomendasi untuk Flutter mobile: INT8 (terkecil)")
    print(f"   Copy ke: project/guidio_app/assets/models/")
    print(f"   Labels : {output_dir}/labels.txt → assets/models/rupiah_labels.txt")

    # Save export summary
    summary = {
        "fp32": {"path": str(fp32_path), "size_mb": size_fp32, "val_accuracy": acc_fp32},
        "fp16": {"path": str(fp16_path), "size_mb": size_fp16, "val_accuracy": acc_fp16},
        "int8": {"path": str(int8_path), "size_mb": size_int8, "val_accuracy": acc_int8},
    }
    with open(output_dir / "export_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Semua model tersimpan di: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
