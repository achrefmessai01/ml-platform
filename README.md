# Strawberry Classifier - YOLO11 Object Detection

A YOLO11-based object detection model for strawberry classification and detection.

## Project Structure

```
strawberry_classifier/
├── data/
│   ├── datasets/            # Raw datasets
│   │   └── farm_dataset/    # Example raw dataset
│   │       ├── images/      # Raw images
│   │       ├── labels/      # Raw labels (YOLO format)
│   │       ├── classes.txt  # Class definitions
│   │       └── notes.json   # Dataset metadata
│   └── processed/           # Processed datasets (train/val/test splits)
│       └── farm_dataset/    # Processed version of farm_dataset
│           ├── train/
│           │   ├── images/  # Training images
│           │   └── labels/  # Training labels
│           ├── val/
│           │   ├── images/  # Validation images
│           │   └── labels/  # Validation labels
│           ├── test/
│           │   ├── images/  # Test images
│           │   └── labels/  # Test labels
│           └── dataset_config.yaml  # Dataset configuration
├── models/                  # Pre-trained and custom models
├── configs/                 # Configuration files
├── scripts/                 # Training and inference scripts
├── utils/                   # Utility functions
├── visualizations/          # Generated plots and visualizations
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

3. Prepare your dataset:
```bash
# Process a raw dataset (e.g., farm_dataset)
python scripts/prepare_data.py --source-dir data/datasets/farm_dataset --dataset-name farm_dataset --train-ratio 0.7 --val-ratio 0.2 --test-ratio 0.1

# Create training configuration
python scripts/dataset_manager.py create-config farm_dataset
```

4. Download YOLO11 pre-trained model:
```bash
# This will be done automatically when you first run training
```

## Dataset Management

### List available datasets:
```bash
python scripts/dataset_manager.py list
```

### Get dataset information:
```bash
python scripts/dataset_manager.py info farm_dataset
```

### Prepare a new dataset:
```bash
python scripts/prepare_data.py --source-dir data/datasets/your_dataset --dataset-name your_dataset --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1
```

### Analyze dataset distribution:
```bash
# Analyze processed dataset
python scripts/analyze_dataset.py --dataset-path data/processed/farm_dataset --dataset-name farm_dataset

# Analyze raw dataset
python scripts/analyze_dataset.py --dataset-path data/datasets/farm_dataset --dataset-name farm_dataset_raw

# Visualizations are saved to the visualizations/ directory
```

## Usage

### Training
```bash
# Train with specific dataset
python scripts/train.py --data configs/farm_dataset_data.yaml

# Train with custom parameters
python scripts/train.py --data configs/farm_dataset_data.yaml --epochs 100 --batch-size 16
```

### Validation
```bash
# Validate with specific dataset
python scripts/validate.py --model runs/train/strawberry_classifier/weights/best.pt --data configs/farm_dataset_data.yaml
```

### Inference
```bash
python scripts/predict.py --model runs/train/strawberry_classifier/weights/best.pt --source path/to/image.jpg
```

## Data Format

This project uses YOLO format for annotations:
- Images: JPG, PNG formats
- Labels: TXT files with same name as image files
- Label format: `class_id x_center y_center width height` (normalized 0-1)

## Classes

The farm_dataset contains the following classes:
- 0: Flower
- 1: Flower_Fruit
- 2: Green_Fruit
- 3: Green_White_Fruit
- 4: Red_Fruit
- 5: UNSURE
- 6: White_Red_Fruit

You can modify class definitions in your dataset's `classes.txt` file or `notes.json` file.

## Data Structure

### Raw Dataset Format
Place your raw datasets in `data/datasets/your_dataset_name/` with the following structure:
- `images/` - All image files
- `labels/` - YOLO format label files (same name as images but with .txt extension)
- `classes.txt` - List of class names (one per line)
- `notes.json` - Optional metadata file

### Processed Dataset Format
After running `prepare_data.py`, datasets are organized into:
- `data/processed/your_dataset_name/train/` - Training split
- `data/processed/your_dataset_name/val/` - Validation split
- `data/processed/your_dataset_name/test/` - Test split

Each split contains `images/` and `labels/` subdirectories.
