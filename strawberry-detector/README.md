# Strawberry Detector

MMDetection-based strawberry detection project with 7 strawberry classes: Flower, Flower_Fruit, Green_Fruit, Green_White_Fruit, Red_Fruit, UNSURE, and White_Red_Fruit.

## Project Structure
```
├── configs/
│   └── strawberry_config.py    # Training configuration (inherits from Faster R-CNN)
├── split_coco_dataset.py       # Split and organize COCO datasets
├── fix_coco_paths.py           # Fix broken image paths in COCO annotations
├── visualize_coco.py           # Visualize COCO dataset samples with annotations
├── train.py                    # Training script
├── test.py                     # Inference script
└── requirements.txt            # Dependencies
```

## Dataset Structure
After processing with `split_coco_dataset.py`, your dataset should look like:
```
dataset_organized/
├── annotations/
│   ├── instances_train.json
│   ├── instances_val.json
│   └── instances_test.json
├── train/                      # Training images
├── val/                        # Validation images
└── test/                       # Test images
```

## Setup

1. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```

## Usage

### 1. Dataset Preparation

**Split your COCO dataset:**
```bash
# Split dataset into train/val/test (70/20/10) and organize images
python split_coco_dataset.py /path/to/your/dataset_directory

# Custom ratios
python split_coco_dataset.py /path/to/dataset --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05
```

**Fix broken annotation paths (if needed):**
```bash
# Preview what would be fixed
python fix_coco_paths.py annotations.json images/ --dry-run

# Fix paths in annotation file
python fix_coco_paths.py annotations.json images/
```

### 2. Training

```bash
python train.py configs/strawberry_config.py
```

### 3. Inference

```bash
python test.py configs/strawberry_config.py work_dirs/strawberry_detector/latest.pth image.jpg --out result.jpg
```

### 4. Dataset Visualization

```bash
# Visualize training sample
python visualize_coco.py configs/strawberry_config.py --split train --idx 0

# Visualize validation sample with custom output
python visualize_coco.py configs/strawberry_config.py --split val --idx 5 --out-file sample.jpg
```

## Configuration

The `strawberry_config.py` uses inheritance from MMDetection's Faster R-CNN base config and modifies:
- **Classes**: 7 custom strawberry classes
- **Dataset paths**: Points to your organized dataset
- **Model head**: Configured for 7-class detection

## Scripts Overview

- **`split_coco_dataset.py`**: Takes a dataset directory and splits it into train/val/test with organized structure
- **`fix_coco_paths.py`**: Fixes broken relative paths in COCO annotation files
- **`visualize_coco.py`**: Visualizes dataset samples with real bounding box annotations
