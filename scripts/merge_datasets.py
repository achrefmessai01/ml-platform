#!/usr/bin/env python3
"""
Simple Dataset Merger

This script merges two YOLO datasets by copying images and labels from both source
datasets into a new combined dataset. It    print(f"\nDataset merge completed successfully!")
    print(f"Total classes in merged dataset: {len(merged_classes)}")
    
    # Print statistics
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_path, split, 'images')
        if os.path.exists(split_dir):
            count = len([f for f in os.listdir(split_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"{split.capitalize()} images: {count}")
    
    return Trueclass remapping if the datasets
have different class names.
"""

import os
import shutil
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Tuple


def load_yaml_config(yaml_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml_config(config: Dict, yaml_path: str):
    """Save YAML configuration file."""
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def detect_label_format(dataset_path: str) -> str:
    """Detect label format (BB or OBB) by examining the first label file found."""
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(dataset_path, split, 'labels')
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
            if label_files:
                # Check only the first label file
                label_path = os.path.join(labels_dir, label_files[0])
                with open(label_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) == 5:
                                return "BB"  # class_id center_x center_y width height
                            elif len(parts) == 9:
                                return "OBB"  # class_id x1 y1 x2 y2 x3 y3 x4 y4
                            else:
                                return "UNKNOWN"
    return "EMPTY"


def get_class_mapping(source_classes: List[str], target_classes: List[str]) -> Dict[int, int]:
    """Create mapping from source class indices to target class indices."""
    mapping = {}
    for i, class_name in enumerate(source_classes):
        if class_name in target_classes:
            mapping[i] = target_classes.index(class_name)
        else:
            # Add new class to target
            target_classes.append(class_name)
            mapping[i] = len(target_classes) - 1
    return mapping


def remap_label_file(label_path: str, class_mapping: Dict[int, int], output_path: str):
    """Remap class indices in a label file and save to output path."""
    if not os.path.exists(label_path):
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    remapped_lines = []
    for line in lines:
        if line.strip():
            parts = line.strip().split()
            old_class_id = int(parts[0])
            new_class_id = class_mapping.get(old_class_id, old_class_id)
            parts[0] = str(new_class_id)
            remapped_lines.append(' '.join(parts) + '\n')
    
    with open(output_path, 'w') as f:
        f.writelines(remapped_lines)


def copy_dataset_split(source_dir: str, target_dir: str, split: str, class_mapping: Dict[int, int]):
    """Copy images and labels for a specific split (train/val/test)."""
    source_images = os.path.join(source_dir, split, 'images')
    source_labels = os.path.join(source_dir, split, 'labels')
    target_images = os.path.join(target_dir, split, 'images')
    target_labels = os.path.join(target_dir, split, 'labels')
    
    # Create target directories
    os.makedirs(target_images, exist_ok=True)
    os.makedirs(target_labels, exist_ok=True)
    
    if not os.path.exists(source_images):
        print(f"Warning: {source_images} does not exist, skipping {split} split")
        return
    
    # Copy images
    for img_file in os.listdir(source_images):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            source_img_path = os.path.join(source_images, img_file)
            target_img_path = os.path.join(target_images, img_file)
            shutil.copy2(source_img_path, target_img_path)
            
            # Copy and remap corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            source_label_path = os.path.join(source_labels, label_file)
            target_label_path = os.path.join(target_labels, label_file)
            
            if os.path.exists(source_label_path):
                remap_label_file(source_label_path, class_mapping, target_label_path)


def merge_datasets(dataset1_path: str, dataset2_path: str, output_path: str):
    """Merge two YOLO datasets into a new combined dataset."""
    
    # Load configurations
    config1 = load_yaml_config(os.path.join(dataset1_path, 'data.yaml'))
    config2 = load_yaml_config(os.path.join(dataset2_path, 'data.yaml'))
    
    # Check label format compatibility
    format1 = detect_label_format(dataset1_path)
    format2 = detect_label_format(dataset2_path)
    
    print(f"Dataset 1 label format: {format1}")
    print(f"Dataset 2 label format: {format2}")
    
    if format1 != format2 and format1 != "EMPTY" and format2 != "EMPTY":
        print(f"ERROR: Label format mismatch!")
        print(f"  Dataset 1 uses {format1} format")
        print(f"  Dataset 2 uses {format2} format")
        print(f"  Both datasets must use the same label format (BB or OBB)")
        return False
    
    # Start with classes from dataset1
    merged_classes = config1['names'].copy()
    
    # Create class mapping for dataset2
    class_mapping = get_class_mapping(config2['names'], merged_classes)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Merging datasets:")
    print(f"  Dataset 1: {dataset1_path}")
    print(f"  Dataset 2: {dataset2_path}")
    print(f"  Output: {output_path}")
    print(f"  Combined classes: {merged_classes}")
    
    # Copy dataset1 (no remapping needed)
    print("\nCopying Dataset 1...")
    for split in ['train', 'val', 'test']:
        copy_dataset_split(dataset1_path, output_path, split, {i: i for i in range(len(config1['names']))})
    
    # Copy dataset2 (with class remapping)
    print("Copying Dataset 2...")
    for split in ['train', 'val', 'test']:
        copy_dataset_split(dataset2_path, output_path, split, class_mapping)
    
    # Create merged configuration
    merged_config = {
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(merged_classes),
        'names': merged_classes
    }
    
    # Save merged configuration
    save_yaml_config(merged_config, os.path.join(output_path, 'data.yaml'))
    
    print(f"\nDataset merge completed successfully!")
    print(f"Total classes in merged dataset: {len(merged_classes)}")
    
    # Print statistics
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(output_path, split, 'images')
        if os.path.exists(split_dir):
            count = len([f for f in os.listdir(split_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))])
            print(f"{split.capitalize()} images: {count}")


def main():
    parser = argparse.ArgumentParser(description='Merge two YOLO datasets')
    parser.add_argument('dataset1', help='Path to first dataset directory')
    parser.add_argument('dataset2', help='Path to second dataset directory')
    parser.add_argument('output', help='Path to output merged dataset directory')
    
    args = parser.parse_args()
    
    # Validate input paths
    if not os.path.exists(args.dataset1):
        print(f"Error: Dataset 1 path does not exist: {args.dataset1}")
        return
    
    if not os.path.exists(args.dataset2):
        print(f"Error: Dataset 2 path does not exist: {args.dataset2}")
        return
    
    if not os.path.exists(os.path.join(args.dataset1, 'data.yaml')):
        print(f"Error: data.yaml not found in dataset 1: {args.dataset1}")
        return
    
    if not os.path.exists(os.path.join(args.dataset2, 'data.yaml')):
        print(f"Error: data.yaml not found in dataset 2: {args.dataset2}")
        return
    
    # Merge datasets
    success = merge_datasets(args.dataset1, args.dataset2, args.output)
    if not success:
        print("Dataset merge failed!")
        return


if __name__ == '__main__':
    main()
