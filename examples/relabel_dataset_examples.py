#!/usr/bin/env python3
"""
Example Usage and Documentation for Dataset Relabeling Script

This file shows various ways to use the relabel_dataset.py script
for different scenarios and requirements.
"""

# =============================================================================
# EXAMPLE 1: Basic relabeling with command line arguments
# =============================================================================
"""
Relabel a dataset using a trained YOLO classifier:

python scripts/relabel_dataset.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/relabeled/farm_dataset_v2 \
    --classifier_model models/strawberry_classifier.pt \
    --confidence_threshold 0.7

This will:
- Load the original dataset from data/processed/farm_dataset
- Crop each labeled object from the images
- Run inference using the strawberry_classifier.pt model
- Create a new dataset with predicted labels
- Only keep predictions with confidence >= 0.7
- Preserve the train/val/test split structure
"""

# =============================================================================
# EXAMPLE 2: Using configuration file
# =============================================================================
"""
For complex setups, use a configuration file:

1. Create config file:
python scripts/relabel_dataset.py --create_config

2. Edit the generated relabel_config.yaml:
"""

config_example = """
dataset_path: 'data/processed/farm_dataset'
output_dir: 'data/relabeled/farm_dataset_v2'
classifier_model: 'models/strawberry_classifier.pt'
confidence_threshold: 0.7
crop_padding: 0.15
min_crop_size: 64
preserve_splits: true
skip_low_confidence: true
"""

"""
3. Run with config:
python scripts/relabel_dataset.py --config relabel_config.yaml
"""

# =============================================================================
# EXAMPLE 3: Conservative relabeling (high confidence only)
# =============================================================================
"""
Only relabel objects with very high confidence:

python scripts/relabel_dataset.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/relabeled/high_confidence \
    --classifier_model models/strawberry_classifier.pt \
    --confidence_threshold 0.9 \
    --crop_padding 0.05

This will:
- Use 90% confidence threshold (very conservative)
- Use minimal padding (5%) for precise crops
- Skip any predictions below 90% confidence
- Good for creating high-quality training data
"""

# =============================================================================
# EXAMPLE 4: Inclusive relabeling (keep low confidence predictions)
# =============================================================================
"""
Keep all predictions regardless of confidence:

python scripts/relabel_dataset.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/relabeled/inclusive \
    --classifier_model models/strawberry_classifier.pt \
    --confidence_threshold 0.1 \
    --no_skip_low_confidence

This will:
- Use very low confidence threshold (10%)
- Keep even uncertain predictions
- Useful for datasets where you want maximum recall
- Good for preliminary analysis or when you trust your classifier
"""

# =============================================================================
# EXAMPLE 5: Fine-tuned preprocessing
# =============================================================================
"""
Use custom preprocessing settings:

python scripts/relabel_dataset.py \
    --dataset_path data/processed/farm_dataset \
    --output_dir data/relabeled/fine_tuned \
    --classifier_model models/strawberry_classifier.pt \
    --confidence_threshold 0.6 \
    --crop_padding 0.2 \
    --min_crop_size 128

This will:
- Use 20% padding around crops (more context)
- Require minimum 128x128 pixel crops
- Skip very small objects that might be noise
- Good for classifiers that need more context
"""

# =============================================================================
# WORKFLOW EXAMPLES
# =============================================================================

# Workflow 1: Improving dataset quality
workflow_1 = """
1. Train initial classifier on your manually labeled data
2. Use relabel_dataset.py to auto-label new data
3. Manually review high-confidence predictions
4. Retrain classifier with expanded dataset
5. Repeat process for iterative improvement
"""

# Workflow 2: Cross-validation of labels
workflow_2 = """
1. Split your dataset into train/val
2. Train classifier on train set
3. Use relabel_dataset.py on val set
4. Compare original vs predicted labels
5. Identify systematic labeling errors
6. Correct and retrain
"""

# Workflow 3: Domain adaptation
workflow_3 = """
1. Train classifier on source domain
2. Use relabel_dataset.py on target domain
3. Review predictions for domain-specific patterns
4. Fine-tune classifier on target domain samples
5. Re-label target domain with improved model
"""

# =============================================================================
# OUTPUT STRUCTURE
# =============================================================================
"""
The script creates this output structure:

data/relabeled/farm_dataset_v2/
├── train/
│   ├── images/           # Original images (copied)
│   └── labels/           # New labels with predicted classes
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml             # Updated with new class names
├── classes.txt           # New class names from classifier
└── relabeling_stats.json # Detailed statistics

The relabeling_stats.json contains:
- Total images and labels processed
- Number of labels relabeled vs skipped
- Original vs new class distributions
- Confidence statistics (min/max/average)
- Processing details
"""

# =============================================================================
# QUALITY CONTROL TIPS
# =============================================================================
"""
1. Start with high confidence thresholds (0.8-0.9)
2. Manually review a sample of predictions
3. Check class distribution changes for reasonableness
4. Use visualization tools to inspect relabeled data
5. Compare original vs predicted labels on validation set
6. Monitor confidence statistics in output

Good confidence threshold guidelines:
- 0.9+: Very conservative, high precision
- 0.7-0.9: Balanced approach, good quality
- 0.5-0.7: More inclusive, review recommended
- <0.5: Very inclusive, manual review required
"""

# =============================================================================
# TROUBLESHOOTING
# =============================================================================
"""
Common issues and solutions:

1. "Classifier model not found"
   - Check the path to your .pt file
   - Make sure the model is trained for classification (not detection)

2. "Low number of relabeled labels"
   - Lower confidence threshold
   - Check if classifier classes match your data
   - Verify crop padding and min size settings

3. "Memory issues with large datasets"
   - Process in smaller batches
   - Use smaller crop sizes
   - Ensure sufficient disk space

4. "Unexpected class predictions"
   - Review classifier training data
   - Check for domain mismatch
   - Visualize crops to verify quality

5. "All predictions skipped"
   - Lower confidence threshold
   - Check min_crop_size setting
   - Verify image and label file alignment
"""

# =============================================================================
# INTEGRATION WITH OTHER SCRIPTS
# =============================================================================
"""
Use with other scripts in this project:

1. After relabeling, extract crops for classification:
python scripts/extract_crops_for_classification.py \
    --dataset_path data/relabeled/farm_dataset_v2 \
    --output_dir data/classification/relabeled_crops

2. Train a new model on relabeled data:
python scripts/train.py \
    --data data/relabeled/farm_dataset_v2/data.yaml \
    --epochs 100

3. Validate relabeling quality:
python scripts/validate.py \
    --data data/relabeled/farm_dataset_v2/data.yaml \
    --model models/original_classifier.pt

4. Compare original vs relabeled datasets:
python scripts/analyze_dataset.py \
    --dataset1 data/processed/farm_dataset \
    --dataset2 data/relabeled/farm_dataset_v2
"""

if __name__ == "__main__":
    print("This file contains examples and documentation.")
    print("Run the actual relabeling script:")
    print("python scripts/relabel_dataset.py --help")
