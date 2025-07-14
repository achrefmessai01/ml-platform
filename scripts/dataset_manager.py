#!/usr/bin/env python3
"""
Dataset management script for handling multiple datasets
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_yaml, save_yaml, validate_dataset_structure

def list_datasets(base_dir: str = "data"):
    """List all available datasets"""
    datasets = {}
    
    # Check raw datasets
    raw_datasets_dir = os.path.join(base_dir, "datasets")
    if os.path.exists(raw_datasets_dir):
        for item in os.listdir(raw_datasets_dir):
            item_path = os.path.join(raw_datasets_dir, item)
            if os.path.isdir(item_path):
                datasets[item] = {
                    'type': 'raw',
                    'path': item_path,
                    'processed': False
                }
    
    # Check processed datasets
    processed_datasets_dir = os.path.join(base_dir, "processed")
    if os.path.exists(processed_datasets_dir):
        for item in os.listdir(processed_datasets_dir):
            item_path = os.path.join(processed_datasets_dir, item)
            if os.path.isdir(item_path):
                if item in datasets:
                    datasets[item]['processed'] = True
                    datasets[item]['processed_path'] = item_path
                else:
                    datasets[item] = {
                        'type': 'processed',
                        'path': item_path,
                        'processed': True
                    }
    
    return datasets

def get_dataset_info(dataset_name: str, base_dir: str = "data"):
    """Get detailed information about a dataset"""
    datasets = list_datasets(base_dir)
    
    if dataset_name not in datasets:
        return None
    
    dataset = datasets[dataset_name]
    info = {
        'name': dataset_name,
        'type': dataset['type'],
        'path': dataset['path'],
        'processed': dataset['processed']
    }
    
    # Get statistics if processed
    if dataset['processed']:
        processed_path = dataset.get('processed_path', dataset['path'])
        stats = validate_dataset_structure(processed_path)
        info['statistics'] = stats
        
        # Load dataset config if available
        config_file = os.path.join(processed_path, 'dataset_config.yaml')
        if os.path.exists(config_file):
            info['config'] = load_yaml(config_file)
    
    # Check raw dataset structure
    if dataset['type'] == 'raw' or 'raw' in dataset:
        raw_path = dataset['path'] if dataset['type'] == 'raw' else dataset.get('raw_path')
        if raw_path:
            info['raw_info'] = analyze_raw_dataset(raw_path)
    
    return info

def analyze_raw_dataset(dataset_path: str):
    """Analyze a raw dataset structure"""
    info = {}
    
    # Check for images directory
    images_dir = os.path.join(dataset_path, 'images')
    if os.path.exists(images_dir):
        from utils.data_utils import get_image_paths
        image_files = get_image_paths(images_dir)
        info['total_images'] = len(image_files)
    
    # Check for labels directory
    labels_dir = os.path.join(dataset_path, 'labels')
    if os.path.exists(labels_dir):
        label_files = list(Path(labels_dir).glob('*.txt'))
        info['total_labels'] = len(label_files)
    
    # Check for classes.txt
    classes_file = os.path.join(dataset_path, 'classes.txt')
    if os.path.exists(classes_file):
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
            info['classes'] = classes
            info['num_classes'] = len(classes)
    
    # Check for notes.json
    notes_file = os.path.join(dataset_path, 'notes.json')
    if os.path.exists(notes_file):
        try:
            with open(notes_file, 'r') as f:
                notes = json.load(f)
                if 'categories' in notes:
                    info['categories'] = notes['categories']
        except:
            pass
    
    return info

def create_training_config(dataset_name: str, base_dir: str = "data", output_file: str | None = None):
    """Create a training configuration file for a specific dataset"""
    datasets = list_datasets(base_dir)
    
    if dataset_name not in datasets:
        print(f"Dataset '{dataset_name}' not found")
        return None
    
    dataset = datasets[dataset_name]
    
    if not dataset['processed']:
        print(f"Dataset '{dataset_name}' is not processed yet. Please run prepare_data.py first.")
        return None
    
    processed_path = dataset.get('processed_path', dataset['path'])
    config_file = os.path.join(processed_path, 'dataset_config.yaml')
    
    if not os.path.exists(config_file):
        print(f"Dataset config file not found: {config_file}")
        return None
    
    # Load dataset config
    dataset_config = load_yaml(config_file)
    
    # Create training config
    training_config = {
        'train': os.path.join(processed_path, 'train', 'images'),
        'val': os.path.join(processed_path, 'val', 'images'),
        'test': os.path.join(processed_path, 'test', 'images'),
        'nc': dataset_config['nc'],
        'names': dataset_config['names']
    }
    
    # Save config within the dataset folder
    if output_file is None:
        output_file = f"data/processed/{dataset_name}/data.yaml"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    save_yaml(training_config, output_file)
    
    print(f"Training configuration created: {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Dataset management utility')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List datasets command
    list_parser = subparsers.add_parser('list', help='List all available datasets')
    list_parser.add_argument('--base-dir', type=str, default='data',
                           help='Base directory for datasets')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get information about a dataset')
    info_parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    info_parser.add_argument('--base-dir', type=str, default='data',
                           help='Base directory for datasets')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create training config for a dataset')
    config_parser.add_argument('dataset_name', type=str, help='Name of the dataset')
    config_parser.add_argument('--base-dir', type=str, default='data',
                             help='Base directory for datasets')
    config_parser.add_argument('--output', type=str, 
                             help='Output file for the config (default: data/processed/{dataset_name}/data.yaml)')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        datasets = list_datasets(args.base_dir)
        if not datasets:
            print("No datasets found")
            return
        
        print("Available datasets:")
        print("-" * 50)
        for name, info in datasets.items():
            status = "✓ Processed" if info['processed'] else "○ Raw only"
            print(f"  {name:<20} [{info['type']:<10}] {status}")
    
    elif args.command == 'info':
        info = get_dataset_info(args.dataset_name, args.base_dir)
        if info is None:
            print(f"Dataset '{args.dataset_name}' not found")
            return
        
        print(f"Dataset: {info['name']}")
        print("-" * 40)
        print(f"Type: {info['type']}")
        print(f"Path: {info['path']}")
        print(f"Processed: {'Yes' if info['processed'] else 'No'}")
        
        if 'raw_info' in info:
            raw_info = info['raw_info']
            print(f"\\nRaw dataset info:")
            print(f"  Total images: {raw_info.get('total_images', 'N/A')}")
            print(f"  Total labels: {raw_info.get('total_labels', 'N/A')}")
            print(f"  Classes: {raw_info.get('num_classes', 'N/A')}")
            if 'classes' in raw_info:
                for i, cls in enumerate(raw_info['classes']):
                    print(f"    {i}: {cls}")
        
        if 'statistics' in info:
            print(f"\\nProcessed dataset statistics:")
            for key, value in info['statistics'].items():
                print(f"  {key}: {value}")
        
        if 'config' in info:
            print(f"\\nDataset configuration:")
            print(f"  Number of classes: {info['config']['nc']}")
            print(f"  Class names: {list(info['config']['names'].values())}")
    
    elif args.command == 'create-config':
        create_training_config(args.dataset_name, args.base_dir, args.output)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
