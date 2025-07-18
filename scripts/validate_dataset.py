#!/usr/bin/env python3
"""
Dataset validation and visualization script
"""

import os
import argparse
from utils.data_utils import validate_dataset_structure, load_yaml, visualize_annotations

def main():
    parser = argparse.ArgumentParser(description='Validate dataset and visualize annotations')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Path to dataset directory')
    parser.add_argument('--config', type=str, default='data/processed/farm_dataset/data.yaml',
                       help='Path to data configuration file')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample annotations')
    parser.add_argument('--num-samples', type=int, default=5,
                       help='Number of samples to visualize')
    
    args = parser.parse_args()
    
    print("Validating dataset structure...")
    stats = validate_dataset_structure(args.data_dir)
    
    print("\nDataset Statistics:")
    print("-" * 40)
    for key, value in stats.items():
        print(f"{key:<15}: {value}")
    
    # Check for missing labels
    missing_labels = []
    for split in ['train', 'val', 'test']:
        img_count = stats.get(f'{split}_images', 0)
        lbl_count = stats.get(f'{split}_labels', 0)
        if img_count != lbl_count:
            missing_labels.append(f"{split}: {img_count - lbl_count} missing labels")
    
    if missing_labels:
        print(f"\nWarning: Missing labels detected!")
        for msg in missing_labels:
            print(f"  {msg}")
    else:
        print(f"\n✓ All images have corresponding labels")
    
    # Load class names from config
    if os.path.exists(args.config):
        config = load_yaml(args.config)
        class_names = config.get('names', [])
        print(f"\nClass Names:")
        if isinstance(class_names, list):
            for class_id, class_name in enumerate(class_names):
                print(f"  {class_id}: {class_name}")
        elif isinstance(class_names, dict):
            for class_id, class_name in class_names.items():
                print(f"  {class_id}: {class_name}")
        else:
            print(f"  Warning: Unexpected class names format: {type(class_names)}")
    else:
        print(f"\nWarning: Config file not found: {args.config}")
        class_names = []
    
    # Visualize samples if requested
    if args.visualize and class_names:
        print(f"\nVisualizing {args.num_samples} sample annotations...")
        
        # Create visualizations directory if it doesn't exist
        visualizations_dir = "visualizations"
        os.makedirs(visualizations_dir, exist_ok=True)
        
        # Get sample images from training set
        train_images_dir = os.path.join(args.data_dir, 'train', 'images')
        train_labels_dir = os.path.join(args.data_dir, 'train', 'labels')
        
        if os.path.exists(train_images_dir):
            from utils.data_utils import get_image_paths
            
            image_paths = get_image_paths(train_images_dir)[:args.num_samples]
            
            # Convert class_names to list format
            if isinstance(class_names, list):
                class_names_list = class_names
            elif isinstance(class_names, dict):
                class_names_list = [class_names.get(i, f'class_{i}') for i in range(len(class_names))]
            else:
                class_names_list = []
            
            for i, image_path in enumerate(image_paths):
                image_name = os.path.basename(image_path)
                label_name = os.path.splitext(image_name)[0] + '.txt'
                label_path = os.path.join(train_labels_dir, label_name)
                
                # Use original image name with validation prefix
                image_stem = os.path.splitext(image_name)[0]
                image_ext = os.path.splitext(image_name)[1]
                output_path = os.path.join(visualizations_dir, f"validation_{image_stem}.jpg")
                
                try:
                    visualize_annotations(image_path, label_path, class_names_list, output_path)
                    print(f"  Saved visualization: {output_path}")
                except Exception as e:
                    print(f"  Error visualizing {image_name}: {str(e)}")

if __name__ == "__main__":
    main()
