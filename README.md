# Strawberry Classifier - YOLO11 Object Detection

A YOLO11-based object detection model for strawberry classification and detection.

## Project Structure

```
strawberry_classifier/
├── data/
│   ├── train/
│   │   ├── images/          # Training images
│   │   └── labels/          # Training labels (YOLO format)
│   ├── val/
│   │   ├── images/          # Validation images
│   │   └── labels/          # Validation labels (YOLO format)
│   └── test/
│       ├── images/          # Test images
│       └── labels/          # Test labels (YOLO format)
├── models/                  # Pre-trained and custom models
├── configs/                 # Configuration files
├── scripts/                 # Training and inference scripts
├── utils/                   # Utility functions
├── requirements.txt         # Project dependencies
└── README.md               # This file
```

## Setup

1. Create venv
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download YOLO11 pre-trained model:
```bash
# This will be done automatically when you first run training
```

## Usage

### Training
```bash
python scripts/train.py
```

### Validation
```bash
python scripts/validate.py
```

### Inference
```bash
python scripts/predict.py --source path/to/image.jpg
```

## Data Format

This project uses YOLO format for annotations:
- Images: JPG, PNG formats
- Labels: TXT files with same name as image files
- Label format: `class_id x_center y_center width height` (normalized 0-1)

## Classes

Define your strawberry classes in `configs/data.yaml`:
- 0: ripe_strawberry
- 1: unripe_strawberry
- 2: damaged_strawberry
(Modify as needed for your specific use case)
