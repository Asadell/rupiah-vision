#!/usr/bin/env python3
"""
00b_preflight_check.py
======================
Verifikasi struktur dataset SEBELUM training dimulai.
Harus exit code 0 sebelum menjalankan 01_train.py.

Usage:
    python scripts/00b_preflight_check.py
    python scripts/00b_preflight_check.py --data path/to/classification
"""

import argparse
import sys
from pathlib import Path

REQUIRED_SPLITS = ["train", "val", "test"]
REQUIRED_CLASSES = ["1000", "2000", "5000", "10000", "20000", "50000", "100000"]
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def check_dataset(data_dir: Path) -> list[str]:
    errors = []

    if not data_dir.exists():
        errors.append(f"❌ Root data dir tidak ada: {data_dir.resolve()}")
        return errors

    for split in REQUIRED_SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            errors.append(f"❌ MISSING split folder: {split_dir}")
            continue

        total_split = 0
        for cls in REQUIRED_CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                errors.append(f"❌ MISSING class folder: {cls_dir}")
                continue

            imgs = [p for p in cls_dir.iterdir() if p.suffix.lower() in IMG_EXTENSIONS]
            count = len(imgs)
            if count == 0:
                errors.append(f"❌ EMPTY: {cls_dir} (0 images)")
            else:
                print(f"  ✅ {split:5s}/{cls:>6s}: {count:>5d} images")
                total_split += count

        if total_split > 0:
            print(f"        → Total {split}: {total_split} images\n")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Pre-flight check dataset sebelum training")
    parser.add_argument("--data", default="data/classification", help="Path ke folder classification dataset")
    args = parser.parse_args()

    data_dir = Path(args.data)

    print("=" * 55)
    print("  PRE-FLIGHT CHECK — Rupiah Vision Dataset")
    print("=" * 55)
    print(f"  Data dir: {data_dir.resolve()}")
    print()

    errors = check_dataset(data_dir)

    print()
    if errors:
        print("=" * 55)
        print("  ❌ GAGAL — Ada masalah yang harus diperbaiki:")
        print("=" * 55)
        for e in errors:
            print(f"  {e}")
        print()
        print("  👉 Jalankan 00_merge_and_crop.py terlebih dahulu.")
        sys.exit(1)
    else:
        print("=" * 55)
        print("  ✅ LOLOS — Dataset siap untuk training!")
        print("=" * 55)
        print()
        print("  Jalankan berikutnya:")
        print("    python scripts/01_train.py --data data/classification \\")
        print("        --output models --epochs 30 --batch-size 32")
        sys.exit(0)


if __name__ == "__main__":
    main()
