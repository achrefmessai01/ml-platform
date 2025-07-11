#!/usr/bin/env python3
"""
Data preparation script for strawberry classifier
"""

import os
import shutil
from pathlib import Path
import argparse
from utils.data_utils import validate_dataset_structure, get_image_paths

def prepare_dataset(source_dir: str, output_dir: str = "data", train_ratio: float = 0.7, 
                   val_ratio: float = 0.2, test_ratio: float = 0.1):
    """
    Prepare dataset by splitting images and labels into train/val/test sets
    
    Args:
        source_dir: Directory containing images and labels
        output_dir: Output directory for organized dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Get all image files
    image_files = get_image_paths(source_dir)
    
    if not image_files:
        print(f"No image files found in {source_dir}")
        return
    
    print(f"Found {len(image_files)} images")
    
    # Calculate split indices
    n_total = len(image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    print(f"Split: {n_train} train, {n_val} val, {n_test} test")
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Split and copy files
    splits = [
        ('train', image_files[:n_train]),
        ('val', image_files[n_train:n_train + n_val]),
        ('test', image_files[n_train + n_val:])
    ]
    
    for split_name, split_files in splits:
        for image_path in split_files:
            image_name = Path(image_path).name
            label_name = Path(image_path).stem + '.txt'
            
            # Copy image
            dst_image = os.path.join(output_dir, split_name, 'images', image_name)
            shutil.copy2(image_path, dst_image)
            
            # Copy label if it exists
            label_path = os.path.join(source_dir, label_name)
            if os.path.exists(label_path):
                dst_label = os.path.join(output_dir, split_name, 'labels', label_name)
                shutil.copy2(label_path, dst_label)
    
    print(f"Dataset prepared in {output_dir}")
    
    # Validate the prepared dataset
    stats = validate_dataset_structure(output_dir)
    print("\nDataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO11 training')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Source directory containing images and labels')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for organized dataset')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of training data')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio of validation data')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Ratio of test data')
    
    args = parser.parse_args()
    
    prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )

if __name__ == "__main__":
    main()
