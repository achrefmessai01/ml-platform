#!/usr/bin/env python3
"""
Minimal script to compare datasets and find how many images from one dataset
are contained in another on a split-by-split basis.

Usage:
    python compare_datasets.py --dataset1 path/to/dataset1 --dataset2 path/to/dataset2
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def get_image_files(dataset_path, split_name):
    """Get all image files from a dataset split."""
    images_dir = Path(dataset_path) / split_name / "images"
    if not images_dir.exists():
        return set()
    
    # Get all image files (common extensions)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = set()
    
    for file_path in images_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            # Use just the filename (without extension) for comparison
            # This handles cases where same image might have different extensions
            image_files.add(file_path.stem)
    
    return image_files


def get_dataset_splits(dataset_path):
    """Detect available splits in a dataset."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return []
    
    # Common split names
    possible_splits = ['train', 'val', 'valid', 'test']
    available_splits = []
    
    for split in possible_splits:
        split_dir = dataset_path / split
        if split_dir.exists() and split_dir.is_dir():
            images_dir = split_dir / "images"
            if images_dir.exists():
                available_splits.append(split)
    
    return available_splits


def compare_datasets(dataset1_path, dataset2_path):
    """Compare two datasets and show overlap statistics."""
    dataset1_path = Path(dataset1_path)
    dataset2_path = Path(dataset2_path)
    
    print(f"Comparing datasets:")
    print(f"  Dataset 1: {dataset1_path}")
    print(f"  Dataset 2: {dataset2_path}")
    print()
    
    # Get available splits for both datasets
    splits1 = get_dataset_splits(dataset1_path)
    splits2 = get_dataset_splits(dataset2_path)
    
    if not splits1:
        print(f"No valid splits found in dataset 1: {dataset1_path}")
        return
    
    if not splits2:
        print(f"No valid splits found in dataset 2: {dataset2_path}")
        return
    
    print(f"Dataset 1 splits: {splits1}")
    print(f"Dataset 2 splits: {splits2}")
    print()
    
    # Compare each combination of splits
    results = defaultdict(dict)
    
    for split1 in splits1:
        images1 = get_image_files(dataset1_path, split1)
        
        for split2 in splits2:
            images2 = get_image_files(dataset2_path, split2)
            
            # Find overlap
            overlap = images1.intersection(images2)
            
            results[split1][split2] = {
                'dataset1_count': len(images1),
                'dataset2_count': len(images2),
                'overlap_count': len(overlap),
                'overlap_percentage_1': (len(overlap) / len(images1) * 100) if images1 else 0,
                'overlap_percentage_2': (len(overlap) / len(images2) * 100) if images2 else 0
            }
    
    # Print results
    print("=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    for split1 in splits1:
        print(f"\nDataset 1 - {split1} split ({results[split1][splits2[0]]['dataset1_count']} images):")
        print("-" * 60)
        
        for split2 in splits2:
            stats = results[split1][split2]
            print(f"  vs Dataset 2 - {split2} split ({stats['dataset2_count']} images):")
            print(f"    • Overlap: {stats['overlap_count']} images")
            print(f"    • {stats['overlap_percentage_1']:.1f}% of Dataset 1 {split1} images")
            print(f"    • {stats['overlap_percentage_2']:.1f}% of Dataset 2 {split2} images")
            print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare datasets and find how many images from one dataset are contained in another on a split-by-split basis."
    )
    parser.add_argument(
        "--dataset1", 
        required=True, 
        help="Path to the first dataset"
    )
    parser.add_argument(
        "--dataset2", 
        required=True, 
        help="Path to the second dataset"
    )
    
    args = parser.parse_args()
    
    try:
        compare_datasets(args.dataset1, args.dataset2)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
