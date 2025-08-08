# Strawberry Detector

Barebones MMDetection fine-tuning project for strawberry detection.

## Structure
```
├── configs/
│   └── strawberry_config.py    # Training configuration
├── data/                       # Place your dataset here
│   ├── train/                  # Training images
│   ├── val/                    # Validation images
│   ├── test/                   # Test images
│   └── annotations/            # COCO format annotations
│       ├── instances_train.json
│       ├── instances_val.json
│       └── instances_test.json
├── work_dirs/                  # Training outputs and checkpoints
├── train.py                    # Training script
├── test.py                     # Inference script
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
