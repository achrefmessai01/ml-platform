#!/usr/bin/env python3
"""
Example Usage and Configuration for Crop Extraction Script

This file shows various ways to use the extract_crops_for_classification.py script
for different scenarios and requirements.
"""

# =============================================================================
# EXAMPLE 1: Basic usage with preserved splits
# =============================================================================
"""
Extract crops from your farm dataset preserving train/val/test splits:

python scripts/extract_crops_for_classification.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/classification/big_dataset/strawberry \
    --padding 0.15 \
    --preserve_splits

This will:
- Process all images in train/val/test splits
- Preserve the original train/validation/test split structure
- Add 15% padding around each bounding box
- Save crops organized by split then class (train/Flower/, val/Red_Fruit/, etc.)
- Maintain original crop sizes

Output structure:
data/classification/big_dataset/strawberry/
├── train/
│   ├── Flower/
│   ├── Red_Fruit/
│   └── ...
├── val/
│   ├── Flower/
│   ├── Red_Fruit/
│   └── ...
└── test/
    ├── Flower/
    ├── Red_Fruit/
    └── ...
"""

# =============================================================================
# EXAMPLE 2: Merged labels for binary classification
# =============================================================================
"""
Create a binary classification dataset (strawberry vs background):

python scripts/extract_crops_for_classification.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/classification/binary_strawberry \
    --merge_labels \
    --merged_label_name strawberry \
    --preserve_splits

This will:
- Merge all strawberry classes (Flower, Red_Fruit, etc.) into one "strawberry" class
- Preserve train/val/test splits
- Perfect for binary classification tasks

Output structure:
data/classification/binary_strawberry/
├── train/strawberry/
├── val/strawberry/
└── test/strawberry/
"""

# =============================================================================
# EXAMPLE 3: Fixed-size crops for CNN training
# =============================================================================
"""
Create uniform 224x224 crops for training a CNN classifier:

python scripts/extract_crops_for_classification.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/classification/resized_224/strawberry \
    --crop_size 224x224 \
    --padding 0.1 \
    --preserve_splits

This will:
- Resize all crops to 224x224 pixels (maintaining aspect ratio)
- Add black padding if needed to reach exact size
- Preserve train/val/test splits
- Perfect for standard CNN architectures (ResNet, EfficientNet, etc.)
"""

# =============================================================================
# EXAMPLE 4: High-quality crops with minimal padding
# =============================================================================
"""
Extract high-quality crops with minimal padding for detailed analysis:

python scripts/extract_crops_for_classification.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/classification/high_quality/strawberry \
    --padding 0.05 \
    --min_crop_size 64 \
    --quality 98 \
    --preserve_splits

This will:
- Use only 5% padding (tight crops)
- Skip very small objects (< 64 pixels)
- Save with 98% JPEG quality
- Preserve train/val/test splits
"""

# =============================================================================
# EXAMPLE 5: Flat structure without splits
# =============================================================================
"""
Create a flat directory structure without preserving splits:

python scripts/extract_crops_for_classification.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/classification/flat_structure \
    --no_preserve_splits

This will:
- Combine all images from train/val/test into one dataset
- Create a flat structure with only class folders
- Useful when you want to create your own train/val split later

Output structure:
data/classification/flat_structure/
├── Flower/
├── Red_Fruit/
├── Green_Fruit/
└── ...
"""

# =============================================================================
# EXAMPLE 6: Process disease dataset with merged labels
# =============================================================================
"""
Extract crops from the strawberry disease dataset with merged labels:

python scripts/extract_crops_for_classification.py \
    --dataset_path data/processed/strawberry_disease_dataset \
    --output_dir data/classification/disease_binary \
    --crop_size 256x256 \
    --padding 0.2 \
    --merge_labels \
    --merged_label_name diseased_strawberry \
    --preserve_splits

This will:
- Merge all disease classes into one "diseased_strawberry" class
- Create 256x256 fixed-size crops
- Preserve train/val/test splits
- Perfect for binary disease detection
"""

# =============================================================================
# EXAMPLE 7: Extract only specific labels
# =============================================================================
"""
Extract only Red_Fruit and Green_Fruit classes, ignoring all others:

python scripts/extract_crops_for_classification.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/classification/specific_labels \
    --include_labels Red_Fruit Green_Fruit \
    --preserve_splits

This will:
- Only extract crops for Red_Fruit and Green_Fruit classes
- Skip all other classes (Flower, White_Red_Fruit, etc.)
- Preserve train/val/test splits
- Useful for creating datasets with only specific classes of interest

Output structure:
data/classification/specific_labels/
├── train/
│   ├── Red_Fruit/
│   └── Green_Fruit/
├── val/
│   ├── Red_Fruit/
│   └── Green_Fruit/
└── test/
    ├── Red_Fruit/
    └── Green_Fruit/
"""

# =============================================================================
# EXAMPLE 8: Extract specific labels and merge them
# =============================================================================
"""
Extract only Red_Fruit and Green_Fruit, then merge them into one class:

python scripts/extract_crops_for_classification.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/classification/ripe_strawberries \
    --include_labels Red_Fruit Green_Fruit \
    --merge_labels \
    --merged_label_name ripe_strawberry \
    --preserve_splits

This will:
- Extract only Red_Fruit and Green_Fruit crops
- Merge them into a single "ripe_strawberry" class
- Skip all other classes
- Perfect for binary classification: ripe vs not-ripe

Output structure:
data/classification/ripe_strawberries/
├── train/ripe_strawberry/
├── val/ripe_strawberry/
└── test/ripe_strawberry/
"""

# =============================================================================
# CONFIGURATION FILE EXAMPLE
# =============================================================================
"""
You can also use a configuration file (recommended for complex setups):

1. Create config file:
python scripts/extract_crops_for_classification.py --create_config

2. Edit the generated crop_extraction_config.yaml:
"""

config_example = """
dataset_path: 'data/processed/farm_dataset'
output_dir: 'data/classification/big_dataset/strawberry'
crop_size: [224, 224]     # [width, height] or null for original size
padding: 0.15             # Extra padding around bbox (0.15 = 15%)
min_crop_size: 32         # Minimum crop size in pixels
save_format: 'JPEG'       # JPEG or PNG
quality: 95               # JPEG quality (1-100)
preserve_splits: true     # Preserve train/val/test splits
merge_labels: false       # Merge all labels into one class
merged_label_name: 'strawberry'  # Name for merged class (when merge_labels=true)
include_labels: null      # List of labels to include (null = all labels, e.g., ['Red_Fruit', 'Green_Fruit'])
"""

"""
3. Run with config:
python scripts/extract_crops_for_classification.py --config crop_extraction_config.yaml
"""

# =============================================================================
# OUTPUT STRUCTURE
# =============================================================================
"""
The script creates this directory structure:

data/classification/big_dataset/strawberry/
├── Flower/                    # 762 crops
│   ├── image1_crop_000.jpg
│   ├── image1_crop_001.jpg
│   └── ...
├── Flower_Fruit/             # 999 crops
├── Green_Fruit/              # 957 crops
├── Green_White_Fruit/        # 838 crops
├── Red_Fruit/                # 625 crops
├── UNSURE/                   # 7 crops
├── White_Red_Fruit/          # 449 crops
└── extraction_stats.json     # Detailed statistics

Each crop is named: {original_image_name}_crop_{index:03d}.jpg
"""

# =============================================================================
# CLASSIFICATION TRAINING TIPS
# =============================================================================
"""
After extracting crops, you can train a classifier using frameworks like:

1. TensorFlow/Keras:
   - Use ImageDataGenerator or tf.data for data loading
   - Point to the classification directory structure
   - Classes are automatically detected from folder names

2. PyTorch:
   - Use ImageFolder dataset class
   - Apply transforms for augmentation
   - Classes mapped from directory names

3. Ultralytics (YOLO classification):
   - Can train classification models on this directory structure
   - Supports various architectures (EfficientNet, ResNet, etc.)

Example directory for training:
data/classification/big_dataset/strawberry/  # Point training script here
"""

if __name__ == "__main__":
    print("This file contains examples and documentation.")
    print("Run the actual extraction script:")
    print("python scripts/extract_crops_for_classification.py --help")
