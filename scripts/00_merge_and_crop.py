"""
00_merge_and_crop.py
====================
Merge 3 YOLO/OBB detection datasets → crop bounding boxes → classification dataset.

Supports both:
  - YOLO bbox format:   class cx cy w h
  - OBB/Polygon format: class x1 y1 x2 y2 x3 y3 x4 y4 [x5 y5]  (normalized 0-1)

Output structure:
    <OUTPUT_DIR>/
        train/
            1000/  2000/  5000/  10000/  20000/  50000/  100000/
        val/
            ...
        test/
            ...

Usage:
    python scripts/00_merge_and_crop.py \
        --datasets /path/to/rf-rupiah-detector /path/to/rf-money-detection-valid /path/to/rf-rupiah-skripsi \
        --output data/classification \
        --val-split 0.15 \
        --test-split 0.05 \
        --min-crop-size 32 \
        --padding 0.05
"""

import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

# ─── Class mapping ─────────────────────────────────────────────────────────────
# All 3 datasets share same index order (verified from data.yaml):
#   0 → dua puluh ribu   → 20000
#   1 → dua ribu         → 2000
#   2 → lima puluh ribu  → 50000
#   3 → lima ribu        → 5000
#   4 → sepuluh ribu     → 10000
#   5 → seratus ribu     → 100000
#   6 → seribu           → 1000

IDX_TO_CLASS = {
    0: "20000",
    1: "2000",
    2: "50000",
    3: "5000",
    4: "10000",
    5: "100000",
    6: "1000",
}

CLASS_ORDER = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ─── Label parsing ─────────────────────────────────────────────────────────────

def parse_label_line(line: str):
    """
    Parse one line from a YOLO/OBB label file.
    Returns (class_idx, x_min, y_min, x_max, y_max) in normalized [0,1].
    Returns None if line is invalid.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        cls = int(parts[0])
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None

    n = len(vals)

    if n == 4:
        # YOLO bbox: cx cy w h
        cx, cy, w, h = vals
        x_min = cx - w / 2
        y_min = cy - h / 2
        x_max = cx + w / 2
        y_max = cy + h / 2

    elif n % 2 == 0:
        # OBB / Polygon: x1 y1 x2 y2 ... xN yN
        xs = vals[0::2]
        ys = vals[1::2]
        x_min = min(xs)
        y_min = min(ys)
        x_max = max(xs)
        y_max = max(ys)

    else:
        return None

    # Clamp to [0, 1]
    x_min = max(0.0, min(1.0, x_min))
    y_min = max(0.0, min(1.0, y_min))
    x_max = max(0.0, min(1.0, x_max))
    y_max = max(0.0, min(1.0, y_max))

    if x_max <= x_min or y_max <= y_min:
        return None

    return cls, x_min, y_min, x_max, y_max


# ─── Crop helper ───────────────────────────────────────────────────────────────

def crop_bbox(img: np.ndarray, x_min, y_min, x_max, y_max, padding: float = 0.05):
    """Crop bounding box from image with optional relative padding."""
    H, W = img.shape[:2]
    pw = (x_max - x_min) * padding
    ph = (y_max - y_min) * padding

    x1 = max(0, int((x_min - pw) * W))
    y1 = max(0, int((y_min - ph) * H))
    x2 = min(W, int((x_max + pw) * W))
    y2 = min(H, int((y_max + ph) * H))

    return img[y1:y2, x1:x2]


# ─── Dataset scanner ───────────────────────────────────────────────────────────

def collect_crops_from_dataset(
    dataset_dir: Path,
    padding: float,
    min_crop_size: int,
) -> list[tuple[str, np.ndarray]]:
    """
    Scan one Roboflow YOLO dataset folder and return list of (class_name, crop_array).
    Looks for images in train/valid/test splits.
    """
    crops = []
    splits = ["train", "valid", "test"]

    for split in splits:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            continue

        img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
        print(f"  [{dataset_dir.name}/{split}] {len(img_files)} gambar")

        for img_path in img_files:
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  ⚠️  Tidak bisa baca: {img_path.name}")
                continue

            with open(lbl_path) as f:
                lines = f.readlines()

            for line in lines:
                parsed = parse_label_line(line)
                if parsed is None:
                    continue

                cls_idx, x_min, y_min, x_max, y_max = parsed

                if cls_idx not in IDX_TO_CLASS:
                    continue

                class_name = IDX_TO_CLASS[cls_idx]
                crop = crop_bbox(img, x_min, y_min, x_max, y_max, padding)

                if crop.size == 0 or crop.shape[0] < min_crop_size or crop.shape[1] < min_crop_size:
                    continue

                crops.append((class_name, crop))

    return crops


# ─── Save splits ───────────────────────────────────────────────────────────────

def save_splits(
    all_crops: list[tuple[str, np.ndarray]],
    output_dir: Path,
    val_split: float,
    test_split: float,
):
    """Shuffle, split, and save crops to train/val/test folders."""
    random.shuffle(all_crops)

    n = len(all_crops)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_test - n_val

    splits = {
        "train": all_crops[:n_train],
        "val":   all_crops[n_train: n_train + n_val],
        "test":  all_crops[n_train + n_val:],
    }

    # Create output dirs
    for split in ["train", "val", "test"]:
        for cls in CLASS_ORDER:
            (output_dir / split / cls).mkdir(parents=True, exist_ok=True)

    counters = {cls: {"train": 0, "val": 0, "test": 0} for cls in CLASS_ORDER}

    for split_name, items in splits.items():
        for class_name, crop in items:
            idx = counters[class_name][split_name]
            out_path = output_dir / split_name / class_name / f"{class_name}_{idx:05d}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            counters[class_name][split_name] += 1

    return counters, n_train, n_val, n_test


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Merge YOLO datasets → classification crops")
    parser.add_argument(
        "--datasets", nargs="+", required=True,
        help="Path ke satu atau lebih folder dataset Roboflow YOLO"
    )
    parser.add_argument(
        "--output", default="data/classification",
        help="Output folder untuk classification dataset (default: data/classification)"
    )
    parser.add_argument("--val-split", type=float, default=0.15, help="Proporsi val set (default: 0.15)")
    parser.add_argument("--test-split", type=float, default=0.05, help="Proporsi test set (default: 0.05)")
    parser.add_argument("--padding", type=float, default=0.05, help="Padding relatif di sekitar bbox (default: 0.05)")
    parser.add_argument("--min-crop-size", type=int, default=32, help="Minimum crop size in pixels (default: 32)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output)
    if output_dir.exists():
        print(f"⚠️  Output dir '{output_dir}' sudah ada. Lanjut tanpa hapus.")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all crops
    all_crops = []
    for ds_path in args.datasets:
        ds_dir = Path(ds_path)
        if not ds_dir.exists():
            print(f"❌ Dataset tidak ditemukan: {ds_dir}")
            continue
        print(f"\n📦 Memproses dataset: {ds_dir.name}")
        crops = collect_crops_from_dataset(ds_dir, args.padding, args.min_crop_size)
        print(f"   → {len(crops)} crop berhasil")
        all_crops.extend(crops)

    print(f"\n✅ Total crop: {len(all_crops)}")

    if len(all_crops) == 0:
        print("❌ Tidak ada crop yang dihasilkan. Periksa path dataset.")
        return

    # Distribusi per kelas sebelum split
    print("\n📊 Distribusi per kelas:")
    class_counts = {}
    for cls, _ in all_crops:
        class_counts[cls] = class_counts.get(cls, 0) + 1
    for cls in CLASS_ORDER:
        print(f"   Rp {int(cls):>7,} : {class_counts.get(cls, 0):>5} gambar")

    # Split dan simpan
    print(f"\n✂️  Splitting: train={1-args.val_split-args.test_split:.0%}, val={args.val_split:.0%}, test={args.test_split:.0%}")
    counters, n_train, n_val, n_test = save_splits(
        all_crops, output_dir, args.val_split, args.test_split
    )

    print(f"\n✅ Dataset tersimpan di: {output_dir.resolve()}")
    print(f"   train: {n_train}  |  val: {n_val}  |  test: {n_test}")
    print("\n📁 Struktur output:")
    for split in ["train", "val", "test"]:
        for cls in CLASS_ORDER:
            n = counters[cls][split]
            if n > 0:
                print(f"   {split}/{cls}: {n}")

    # Save class labels file
    labels_file = output_dir / "labels.txt"
    with open(labels_file, "w") as f:
        for cls in CLASS_ORDER:
            f.write(f"{cls}\n")
    print(f"\n📄 Label file: {labels_file}")


if __name__ == "__main__":
    main()
