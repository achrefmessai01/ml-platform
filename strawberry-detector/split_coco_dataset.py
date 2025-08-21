#!/usr/bin/env python3

"""
Split COCO dataset into train/val/test splits and organize images.
"""

import json
import argparse
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path


def split_coco_dataset(dataset_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, seed=42):
    """Split COCO dataset into train/val/test splits and organize images."""
    
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    # Setup paths
    dataset_path = Path(dataset_dir)
    ann_file = dataset_path / 'result.json'  # Standard COCO annotation file
    images_dir = dataset_path / 'images'     # Standard images directory
    
    if not ann_file.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_file}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Load data
    with open(ann_file, 'r') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    
    print(f"Dataset: {dataset_dir}")
    print(f"Total images: {len(images)}, annotations: {len(annotations)}")
    
    # Group annotations by image_id
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    # Shuffle and split image IDs
    image_ids = list(img_to_anns.keys())
    random.seed(seed)
    random.shuffle(image_ids)
    
    total = len(image_ids)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    splits = {
        'train': set(image_ids[:train_size]),
        'val': set(image_ids[train_size:train_size + val_size]),
        'test': set(image_ids[train_size + val_size:])
    }
    
    print(f"Split sizes - Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")
    
    # Create output directory alongside the dataset
    output_path = dataset_path.parent / f"{dataset_path.name}_organized"
    
    # Process each split
    for split_name, img_ids in splits.items():
        if not img_ids:
            continue
            
        # Create directories
        split_dir = output_path / split_name
        ann_dir = output_path / 'annotations'
        split_dir.mkdir(parents=True, exist_ok=True)
        ann_dir.mkdir(parents=True, exist_ok=True)
        
        # Filter and copy images
        split_images = []
        copied = 0
        
        for img in images:
            if img['id'] in img_ids:
                split_images.append(img)
                
                # Copy image
                source_file = images_dir / img['file_name']
                target_file = split_dir / img['file_name']
                
                if source_file.exists():
                    shutil.copy2(source_file, target_file)
                    copied += 1
        
        # Filter annotations
        split_annotations = []
        for img_id in img_ids:
            split_annotations.extend(img_to_anns[img_id])
        
        # Save annotations
        split_data = {
            'info': data.get('info', {}),
            'licenses': data.get('licenses', []),
            'categories': data['categories'],
            'images': split_images,
            'annotations': split_annotations
        }
        
        ann_file_out = ann_dir / f'instances_{split_name}.json'
        with open(ann_file_out, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"{split_name}: {len(split_images)} images, {len(split_annotations)} annotations, {copied} files copied")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Split COCO dataset')
    parser.add_argument('dataset_dir', help='Dataset directory containing result.json and images/')
    parser.add_argument('--train-ratio', type=float, default=0.7, help='Training ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.2, help='Validation ratio (default: 0.2)')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_dir):
        print(f"Error: {args.dataset_dir} not found")
        return
    
    try:
        output_dir = split_coco_dataset(
            args.dataset_dir,
            args.train_ratio,
            args.val_ratio,
            args.test_ratio,
            args.seed
        )
        print(f"✅ Dataset split successfully in {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
