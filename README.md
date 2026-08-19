# Rupiah Vision — Training Pipeline

Pipeline klasifikasi uang Rupiah menggunakan MobileNetV2 → TFLite.

## Struktur Project

```
rupiah-vision/
├── scripts/
│   ├── 00_merge_and_crop.py   # Merge 3 dataset → crop bbox → classification folder
│   ├── 01_train.py            # Training MobileNetV2 (2-stage fine-tuning)
│   └── 02_export_tflite.py    # Export → FP32 / FP16 / INT8 TFLite
├── data/
│   └── classification/        # Output dari 00_ (dibuat otomatis)
│       ├── train/ val/ test/
│       └── 1000/ 2000/ 5000/ 10000/ 20000/ 50000/ 100000/
├── models/                    # Output dari 01_ dan 02_
│   ├── rupiah_mobilenetv2/    # SavedModel
│   ├── tflite/
│   │   ├── rupiah_classifier.tflite
│   │   ├── rupiah_classifier_fp16.tflite
│   │   └── rupiah_classifier_int8.tflite   ← deploy ke Flutter
│   └── training_history.png
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

> **GPU server:** install tensorflow-gpu dan pastikan CUDA/cuDNN compatible

## Langkah Training

### Step 0 — Merge & Crop Dataset

Ubah path `--datasets` sesuai lokasi dataset kamu:

```bash
python scripts/00_merge_and_crop.py \
    --datasets \
        ~/datasets/rupiah-detection/rf-rupiah-detector \
        ~/datasets/rupiah-detection/rf-money-detection-valid \
        ~/datasets/rupiah-detection/rf-rupiah-skripsi \
    --output data/classification \
    --val-split 0.15 \
    --test-split 0.05 \
    --padding 0.05
```

Output: folder `data/classification/` dengan 7 subfolder per split.

### Step 1 — Training

```bash
python scripts/01_train.py \
    --data data/classification \
    --output models \
    --epochs 30 \
    --unfreeze-epochs 20 \
    --batch-size 32
```

- **Tahap 1 (30 epoch):** Training head classifier saja, base frozen, LR=1e-3
- **Tahap 2 (20 epoch):** Fine-tune 80 layer terakhir MobileNetV2, LR=1e-5
- Best model disimpan otomatis (`models/best_stage2.keras`)

### Step 2 — Export TFLite

```bash
python scripts/02_export_tflite.py \
    --model models/rupiah_mobilenetv2 \
    --data data/classification \
    --output models/tflite \
    --rep-samples 200
```

Output: 3 file `.tflite` + `labels.txt` + `export_summary.json`

## Deploy ke Flutter

Copy file ke project:
```
models/tflite/rupiah_classifier_int8.tflite → assets/models/uang_rupiah.tflite
models/tflite/labels.txt                    → assets/models/rupiah_labels.txt
```

## Kelas (7 Kelas)

| idx | Folder | Nominal |
|-----|--------|---------|
| 0 | `1000`   | Rp 1.000 |
| 1 | `2000`   | Rp 2.000 |
| 2 | `5000`   | Rp 5.000 |
| 3 | `10000`  | Rp 10.000 |
| 4 | `20000`  | Rp 20.000 |
| 5 | `50000`  | Rp 50.000 |
| 6 | `100000` | Rp 100.000 |

## Detail Dataset

| Dataset | Gambar | Format Label |
|---------|--------|--------------|
| rf-rupiah-detector | 1.941 | Polygon/OBB |
| rf-money-detection-valid | 9.101 | OBB (4 corner) |
| rf-rupiah-skripsi | 608 | YOLO bbox |
| **Total** | **~11.650** | - |

Setelah crop bbox, estimasi total crop: **~15.000–20.000** gambar
(1 gambar bisa mengandung beberapa bounding box)
