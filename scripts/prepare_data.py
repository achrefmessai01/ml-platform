#!/usr/bin/env python3
"""
Data preparation script for strawberry classifier
Supports multiple datasets and custom train/val/test splits
"""

import os
import sys
import shutil
import json
import random
from pathlib import Path
import argparse

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import validate_dataset_structure, get_image_paths

def prepare_dataset_from_raw(source_dir: str, dataset_name: str, output_base_dir: str = "data/processed", 
                           train_ratio: float = 0.7, val_ratio: float = 0.2, test_ratio: float = 0.1,
                           seed: int = 42):
    """
    Prepare dataset from raw format (images and labels in separate folders)
    
    Args:
        source_dir: Directory containing raw dataset (with images/ and labels/ subdirs)
        dataset_name: Name of the dataset
        output_base_dir: Base directory for processed datasets
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for reproducibility
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Paths for source data
    images_dir = os.path.join(source_dir, 'images')
    labels_dir = os.path.join(source_dir, 'labels')
    
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(labels_dir):
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    # Get all image files
    image_files = get_image_paths(images_dir)
    
    if not image_files:
        raise ValueError(f"No image files found in {images_dir}")
    
    # Filter images that have corresponding labels
    valid_image_files = []
    for image_path in image_files:
        image_name = Path(image_path).name
        label_name = Path(image_path).stem + '.txt'
        label_path = os.path.join(labels_dir, label_name)
        
        if os.path.exists(label_path):
            valid_image_files.append(image_path)
        else:
            print(f"Warning: No label found for {image_name}, skipping...")
    
    if not valid_image_files:
        raise ValueError("No valid image-label pairs found")
    
    # Shuffle the data
    random.shuffle(valid_image_files)
    
    print(f"Found {len(valid_image_files)} valid image-label pairs")
    
    # Calculate split indices
    n_total = len(valid_image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    print(f"Dataset split: {n_train} train, {n_val} val, {n_test} test")
    
    # Create output directories
    output_dir = os.path.join(output_base_dir, dataset_name)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, split, 'labels'), exist_ok=True)
    
    # Split and copy files
    splits = [
        ('train', valid_image_files[:n_train]),
        ('val', valid_image_files[n_train:n_train + n_val]),
        ('test', valid_image_files[n_train + n_val:])
    ]
    
    for split_name, split_files in splits:
        print(f"Processing {split_name} split...")
        for image_path in split_files:
            image_name = Path(image_path).name
            label_name = Path(image_path).stem + '.txt'
            label_path = os.path.join(labels_dir, label_name)
            
            # Copy image
            dst_image = os.path.join(output_dir, split_name, 'images', image_name)
            shutil.copy2(image_path, dst_image)
            
            # Copy label
            dst_label = os.path.join(output_dir, split_name, 'labels', label_name)
            shutil.copy2(label_path, dst_label)
    
    print(f"Dataset '{dataset_name}' prepared in {output_dir}")
    
    # Copy additional files (classes.txt, notes.json, etc.)
    additional_files = ['classes.txt', 'notes.json', 'README.md']
    for file_name in additional_files:
        src_file = os.path.join(source_dir, file_name)
        if os.path.exists(src_file):
            dst_file = os.path.join(output_dir, file_name)
            shutil.copy2(src_file, dst_file)
            print(f"Copied {file_name} to {dst_file}")
    
    # Create dataset configuration
    create_dataset_config(output_dir, dataset_name, source_dir)
    
    # Validate the prepared dataset
    stats = validate_dataset_structure(output_dir)
    print(f"\nDataset '{dataset_name}' statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    return output_dir

def create_dataset_config(dataset_dir: str, dataset_name: str, source_dir: str):
    """Create a configuration file for the dataset"""
    
    # Load class information
    classes_file = os.path.join(source_dir, 'classes.txt')
    notes_file = os.path.join(source_dir, 'notes.json')
    
    class_names = {}
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
            class_names = {i: cls for i, cls in enumerate(classes)}
    
    # Try to load from notes.json if available
    if os.path.exists(notes_file):
        try:
            with open(notes_file, 'r') as f:
                notes = json.load(f)
                if 'categories' in notes:
                    class_names = {cat['id']: cat['name'] for cat in notes['categories']}
        except:
            pass
    
    # Create dataset config
    config = {
        'name': dataset_name,
        'path': dataset_dir,
        'train': os.path.join(dataset_dir, 'train', 'images'),
        'val': os.path.join(dataset_dir, 'val', 'images'),
        'test': os.path.join(dataset_dir, 'test', 'images'),
        'nc': len(class_names),
        'names': class_names
    }
    
    # Save config
    config_file = os.path.join(dataset_dir, 'dataset_config.yaml')
    from utils.data_utils import save_yaml
    save_yaml(config, config_file)
    print(f"Created dataset configuration: {config_file}")

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for YOLO11 training')
    parser.add_argument('--source-dir', type=str, required=True,
                       help='Source directory containing raw dataset')
    parser.add_argument('--dataset-name', type=str, required=True,
                       help='Name of the dataset')
    parser.add_argument('--output-dir', type=str, default='data/processed',
                       help='Output base directory for processed datasets')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Ratio of training data (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Ratio of validation data (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1,
                       help='Ratio of test data (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"Error: Train ({args.train_ratio}) + Val ({args.val_ratio}) + Test ({args.test_ratio}) = {total_ratio}")
        print("Ratios must sum to 1.0")
        return
    
    try:
        prepare_dataset_from_raw(
            source_dir=args.source_dir,
            dataset_name=args.dataset_name,
            output_base_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        print(f"\n✓ Dataset '{args.dataset_name}' successfully prepared!")
        print(f"  Output directory: {os.path.join(args.output_dir, args.dataset_name)}")
        print(f"  Split ratios: Train={args.train_ratio}, Val={args.val_ratio}, Test={args.test_ratio}")
        print(f"  Random seed: {args.seed}")
        
    except Exception as e:
        print(f"Error preparing dataset: {str(e)}")
        return 1

if __name__ == "__main__":
    main()
