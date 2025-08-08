# Strawberry Detector

Barebones MMDetection fine-tuning project for strawberry detection.

## Structure
```
├── configs/
│   └── strawberry_config.py    # Training configuration (MMDetection 3.x compatible)
├── data/                       # Place your dataset here
│   ├── train/                  # Training images
│   ├── val/                    # Validation images
│   ├── test/                   # Test images
│   └── annotations/            # COCO format annotations
│       ├── instances_train.json
│       ├── instances_val.json
│       └── instances_test.json
├── work_dirs/                  # Training outputs and checkpoints
├── train.py                    # Training script (MMDetection 3.x compatible)
├── test.py                     # Inference script
├── visualize_coco.py           # COCO dataset visualization script
└── requirements.txt            # Dependencies
```

## Usage

1. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt

mim install mmengine

mim install "mmcv>=2.0.0"

mim install mmdet
```

3. Prepare your dataset in COCO format in the `data/` directory

4. Train the model:
```bash
python train.py configs/strawberry_config.py
```

5. Test inference:
```bash
python test.py configs/strawberry_config.py work_dirs/strawberry_detector/latest.pth image.jpg --out result.jpg
```

6. Visualize dataset samples:
```bash
python visualize_coco.py configs/strawberry_config.py --split val --idx 0 --out-file sample_viz.jpg
```

## Dataset Visualization

The `visualize_coco.py` script allows you to visualize samples from your COCO dataset with bounding boxes:

```bash
# Visualize a specific sample
python visualize_coco.py configs/strawberry_config.py --split val --idx 5 --out-file sample.jpg

# Options:
# --split: train, val, or test
# --idx: sample index to visualize
# --out-file: output file path (optional)
```

The script is compatible with both MMDetection 2.x and 3.x formats and will automatically handle path differences between annotation files and actual images.
