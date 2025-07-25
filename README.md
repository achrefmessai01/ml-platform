# Strawberry Classifier

A clean, modular YOLO11-based strawberry detection and classification system.

## Installation

```bash
pip install -e .
```

## Usage

### Command Line Interface

The package provides a unified CLI through the `strawberry-cli` command:

#### Prepare Dataset
```bash
strawberry-cli prepare-data \
    --source-dir data/raw/farm_dataset \
    --output-dir data/processed/farm_dataset \
    --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1
```

#### Extract Crops for Classification
```bash
strawberry-cli extract-crops \
    --dataset-path data/processed/farm_dataset \
    --output-dir data/classification/farm_dataset \
    --crop-width 224 --crop-height 224 \
    --padding 0.1
```

#### Train Detection Model
```bash
strawberry-cli train \
    --model yolo11n.pt \
    --data data/processed/farm_dataset/data.yaml \
    --task detect \
    --epochs 100 \
    --batch-size 16
```

#### Train Classification Model
```bash
strawberry-cli train \
    --model yolo11n-cls.pt \
    --data data/classification/farm_dataset \
    --task classify \
    --epochs 100 \
    --batch-size 16
```

#### Run Prediction
```bash
strawberry-cli predict \
    --model runs/train/exp/weights/best.pt \
    --source data/test_images \
    --task detect \
    --save
```

#### Validate Model
```bash
strawberry-cli validate \
    --model runs/train/exp/weights/best.pt \
    --data data/processed/farm_dataset/data.yaml \
    --task detect
```

### Python API

```python
from strawberry_classifier import StrawberryDetector, StrawberryClassifier

# Detection
detector = StrawberryDetector('path/to/model.pt')
results = detector.predict('path/to/image.jpg')

# Classification  
classifier = StrawberryClassifier('path/to/model.pt')
results = classifier.predict('path/to/image.jpg')
```

## Project Structure

```
strawberry_classifier/
├── strawberry_classifier/           # Main package
│   ├── core/                       # Core models
│   │   ├── detector.py            # Detection model wrapper
│   │   └── classifier.py          # Classification model wrapper
│   ├── data/                      # Data processing
│   │   ├── utils.py              # Utility functions
│   │   ├── prepare.py            # Dataset preparation
│   │   └── crop_extractor.py     # Crop extraction
│   ├── config/                   # Configuration management
│   │   └── config.py            # Config classes
│   └── cli.py                   # Command line interface
├── data/                        # Data directory
│   ├── raw/                    # Raw datasets
│   ├── processed/              # Processed datasets
│   └── classification/         # Classification datasets
├── models/                     # Model storage
├── runs/                      # Training/prediction outputs
└── setup.py                  # Package installation
```

## Data Format

### Detection Dataset
```
dataset/
├── train/
│   ├── images/
│   └── labels/  # YOLO format
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml    # Dataset configuration
```

### Classification Dataset
```
dataset/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
├── val/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
```

## Requirements

- Python >= 3.8
- ultralytics
- opencv-python
- PyYAML
- matplotlib
- tqdm
- numpy
- Pillow
