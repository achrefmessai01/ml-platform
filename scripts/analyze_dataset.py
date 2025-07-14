#!/usr/bin/env python3
"""
Dataset analysis script for visualizing label distribution
"""

import os
import sys
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_yaml, get_image_paths


def analyze_label_distribution(dataset_path: str, dataset_name: str = None):
    """
    Analyze the distribution of labels in a dataset
    
    Args:
        dataset_path: Path to the dataset (processed or raw)
        dataset_name: Name of the dataset for the plot title
    """
    
    # Check if it's a processed dataset or raw dataset
    if os.path.exists(os.path.join(dataset_path, 'train')):
        # Processed dataset
        splits = ['train', 'val', 'test']
        is_processed = True
    else:
        # Raw dataset
        splits = ['']  # Empty string for root directory
        is_processed = False
    
    # Load class names
    class_names = load_class_names(dataset_path)
    
    # Count labels for each split
    split_counts = {}
    total_counts = Counter()
    
    for split in splits:
        if split:
            labels_dir = os.path.join(dataset_path, split, 'labels')
            split_name = split
        else:
            labels_dir = os.path.join(dataset_path, 'labels')
            split_name = 'all'
        
        if not os.path.exists(labels_dir):
            print(f"Warning: Labels directory not found: {labels_dir}")
            continue
        
        # Get all label files
        label_files = list(Path(labels_dir).glob('*.txt'))
        
        # Count labels in this split
        split_counter = Counter()
        
        for label_file in label_files:
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            class_id = int(line.split()[0])
                            split_counter[class_id] += 1
                            total_counts[class_id] += 1
            except (ValueError, IndexError) as e:
                print(f"Warning: Error reading {label_file}: {e}")
                continue
        
        split_counts[split_name] = split_counter
        print(f"\n{split_name.capitalize()} split:")
        print(f"  Files: {len(label_files)}")
        print(f"  Total labels: {sum(split_counter.values())}")
    
    # Create visualizations
    create_distribution_plots(split_counts, total_counts, class_names, dataset_name, is_processed)
    
    # Print summary statistics
    print_summary_statistics(split_counts, total_counts, class_names)


def load_class_names(dataset_path: str) -> dict:
    """Load class names from various sources"""
    class_names = {}
    
    # Try to load from dataset_config.yaml
    config_file = os.path.join(dataset_path, 'dataset_config.yaml')
    if os.path.exists(config_file):
        try:
            config = load_yaml(config_file)
            if 'names' in config:
                class_names = config['names']
                return class_names
        except:
            pass
    
    # Try to load from classes.txt
    classes_file = os.path.join(dataset_path, 'classes.txt')
    if os.path.exists(classes_file):
        try:
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
                class_names = {i: cls for i, cls in enumerate(classes)}
                return class_names
        except:
            pass
    
    # Try to load from notes.json
    notes_file = os.path.join(dataset_path, 'notes.json')
    if os.path.exists(notes_file):
        try:
            with open(notes_file, 'r') as f:
                notes = json.load(f)
                if 'categories' in notes:
                    class_names = {cat['id']: cat['name'] for cat in notes['categories']}
                    return class_names
        except:
            pass
    
    return class_names


def create_distribution_plots(split_counts: dict, total_counts: Counter, class_names: dict, 
                            dataset_name: str, is_processed: bool):
    """Create and save distribution plots"""
    
    # Prepare data for plotting
    if not class_names:
        max_class = max(total_counts.keys()) if total_counts else 0
        class_names = {i: f'Class_{i}' for i in range(max_class + 1)}
    
    # Get all class IDs and sort them
    all_class_ids = sorted(set(total_counts.keys()) | set(class_names.keys()))
    class_labels = [class_names.get(cid, f'Class_{cid}') for cid in all_class_ids]
    
    if is_processed:
        # Create subplot for processed dataset (train/val/test)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Overall distribution
        total_values = [total_counts.get(cid, 0) for cid in all_class_ids]
        bars1 = ax1.bar(class_labels, total_values, color='skyblue', alpha=0.7)
        ax1.set_title(f'Overall Label Distribution - {dataset_name or "Dataset"}')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Labels')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, total_values):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(total_values),
                        str(value), ha='center', va='bottom')
        
        # Plot 2: Split comparison
        splits = ['train', 'val', 'test']
        split_data = []
        split_labels = []
        
        for split in splits:
            if split in split_counts:
                split_data.append([split_counts[split].get(cid, 0) for cid in all_class_ids])
                split_labels.append(split.capitalize())
        
        if split_data:
            x = np.arange(len(class_labels))
            width = 0.25
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for i, (data, label) in enumerate(zip(split_data, split_labels)):
                ax2.bar(x + i * width, data, width, label=label, color=colors[i % len(colors)], alpha=0.7)
            
            ax2.set_title(f'Label Distribution by Split - {dataset_name or "Dataset"}')
            ax2.set_xlabel('Classes')
            ax2.set_ylabel('Number of Labels')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(class_labels, rotation=45)
            ax2.legend()
    
    else:
        # Create single plot for raw dataset
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        
        total_values = [total_counts.get(cid, 0) for cid in all_class_ids]
        bars1 = ax1.bar(class_labels, total_values, color='lightcoral', alpha=0.7)
        ax1.set_title(f'Label Distribution - {dataset_name or "Dataset"}')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Number of Labels')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, total_values):
            if value > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 * max(total_values),
                        str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Create visualizations directory if it doesn't exist
    viz_dir = "visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save plot
    output_filename = os.path.join(viz_dir, f'label_distribution_{dataset_name or "dataset"}.png')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved as: {output_filename}")
    
    # Show plot
    plt.show()


def print_summary_statistics(split_counts: dict, total_counts: Counter, class_names: dict):
    """Print summary statistics"""
    
    print(f"\n{'='*60}")
    print("DATASET SUMMARY STATISTICS")
    print(f"{'='*60}")
    
    total_labels = sum(total_counts.values())
    num_classes = len(total_counts)
    
    print(f"Total labels: {total_labels}")
    print(f"Number of classes: {num_classes}")
    print(f"Average labels per class: {total_labels/num_classes:.1f}")
    
    # Class distribution
    print(f"\nClass distribution:")
    for class_id in sorted(total_counts.keys()):
        count = total_counts[class_id]
        percentage = (count / total_labels) * 100
        class_name = class_names.get(class_id, f'Class_{class_id}')
        print(f"  {class_id}: {class_name:<20} {count:>6} ({percentage:>5.1f}%)")
    
    # Balance analysis
    if num_classes > 1:
        counts = list(total_counts.values())
        min_count = min(counts)
        max_count = max(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\nBalance Analysis:")
        print(f"  Most common class: {max_count} labels")
        print(f"  Least common class: {min_count} labels")
        print(f"  Imbalance ratio: {imbalance_ratio:.2f}:1")
        
        if imbalance_ratio > 10:
            print("  ⚠️  WARNING: Highly imbalanced dataset detected!")
        elif imbalance_ratio > 3:
            print("  ⚠️  WARNING: Moderately imbalanced dataset detected!")
        else:
            print("  ✅ Dataset appears reasonably balanced")


def main():
    parser = argparse.ArgumentParser(description='Analyze dataset label distribution')
    parser.add_argument('--dataset-path', type=str, required=True,
                       help='Path to dataset directory (processed or raw)')
    parser.add_argument('--dataset-name', type=str,
                       help='Name of the dataset for plot titles')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return 1
    
    # Determine dataset name if not provided
    if not args.dataset_name:
        args.dataset_name = os.path.basename(args.dataset_path.rstrip('/'))
    
    try:
        analyze_label_distribution(args.dataset_path, args.dataset_name)
        print(f"\n✅ Analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {str(e)}")
        return 1


if __name__ == "__main__":
    main()
