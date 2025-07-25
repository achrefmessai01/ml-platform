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

#### Visualize Oriented Bounding Boxes
```bash
strawberry-cli visualize-obb \
    --dataset data/processed/farm_dataset_obb \
    --split train \
    --num-samples 6 \
    --save \
    --output obb_visualization.png
```

### Python API

```python
from strawberry_classifier import StrawberryDetector, StrawberryClassifier
from strawberry_classifier.visualization.plot_utils import (
    plot_yolo_obb, 
    visualize_dataset_obb_samples
)

# Detection
detector = StrawberryDetector('path/to/model.pt')
results = detector.predict('path/to/image.jpg')

# Classification  
classifier = StrawberryClassifier('path/to/model.pt')
results = classifier.predict('path/to/image.jpg')

# Visualize oriented bounding boxes
plot_yolo_obb(
    image_path='path/to/image.jpg',
    label_path='path/to/label.txt',
    show_labels=True
)

# Visualize dataset samples
visualize_dataset_obb_samples(
    dataset_path='data/processed/farm_dataset_obb',
    num_samples=6,
    split='train'
)
```

## Visualization Features

The strawberry classifier includes comprehensive visualization tools for oriented bounding box (OBB) datasets and detection results.

### Oriented Bounding Box Visualization

#### Dataset Visualization
Visualize random samples from an OBB dataset to inspect data quality and annotation accuracy:

```python
from strawberry_classifier.visualization.plot_utils import visualize_dataset_obb_samples

# Visualize 6 random samples from the training set
visualize_dataset_obb_samples(
    dataset_path='data/processed/farm_dataset_obb',
    num_samples=6,
    split='train',
    figsize=(15, 10)
)
```

#### Individual Image Visualization
Visualize oriented bounding boxes on a single image:

```python
from strawberry_classifier.visualization.plot_utils import plot_yolo_obb

plot_yolo_obb(
    image_path='path/to/image.jpg',
    label_path='path/to/label.txt',
    class_names=['healthy', 'diseased'],
    show_labels=True,
    line_thickness=2,
    alpha=0.8
)
```

#### Detection Results Visualization
Visualize model detection results with oriented bounding boxes:

```python
from strawberry_classifier.visualization.plot_utils import plot_detections_with_obb

# Assuming you have detection results in the format:
# [{'class_id': 0, 'confidence': 0.95, 'obb_coords': [x1,y1,x2,y2,x3,y3,x4,y4]}]
plot_detections_with_obb(
    image_path='path/to/image.jpg',
    detections=detection_results,
    class_names=['healthy', 'diseased'],
    confidence_threshold=0.5,
    show_confidence=True
)
```

#### Command Line Visualization
Use the CLI to quickly visualize OBB datasets:

```bash
# Display visualization
strawberry-cli visualize-obb --dataset data/processed/farm_dataset_obb --split train

# Save visualization to file
strawberry-cli visualize-obb \
    --dataset data/processed/farm_dataset_obb \
    --split val \
    --num-samples 9 \
    --save \
    --output validation_samples.png \
    --width 18 \
    --height 12
```

### YOLO OBB Format

The visualization tools support the YOLO oriented bounding box format where each line in a label file contains:
```
class_id x1 y1 x2 y2 x3 y3 x4 y4
```

- `class_id`: Integer class identifier
- `x1,y1` to `x4,y4`: Normalized coordinates (0-1) of the four corners of the oriented rectangle

The coordinates represent the four corners of the rotated bounding box in order, allowing for accurate visualization of objects at any orientation.

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
│   ├── visualization/            # Visualization tools
│   │   └── plot_utils.py        # OBB plotting functions
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

### Optional Dependencies for Enhanced Visualization
- seaborn (for improved color palettes)
- plotly (for interactive visualizations)
