#!/usr/bin/env python3
"""
Create a subset of a dataset by filtering specific label classes
Supports YOLO format datasets with train/val/test splits
"""

import os
import sys
import json
import shutil
import argparse
import numpy as np
from pathlib import Path
from typing import List, Set, Dict, Tuple

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_yaml, save_yaml, validate_dataset_structure

# Import Ultralytics utility for segment to bbox conversion
try:
    from ultralytics.utils.ops import segments2boxes
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    segments2boxes = None
    ULTRALYTICS_AVAILABLE = False

def load_class_names(dataset_dir: str) -> Dict[int, str]:
    """Load class names from dataset configuration"""
    data_yaml_path = os.path.join(dataset_dir, 'data.yaml')
    classes_txt_path = os.path.join(dataset_dir, 'classes.txt')
    notes_json_path = os.path.join(dataset_dir, 'notes.json')
    
    class_names = {}
    
    # Try data.yaml first (YOLO format)
    if os.path.exists(data_yaml_path):
        try:
            config = load_yaml(data_yaml_path)
            if 'names' in config:
                if isinstance(config['names'], list):
                    class_names = {i: name for i, name in enumerate(config['names'])}
                elif isinstance(config['names'], dict):
                    class_names = {int(k): v for k, v in config['names'].items()}
        except Exception as e:
            print(f"Warning: Could not load classes from data.yaml: {e}")
    
    # Fallback to classes.txt
    if not class_names and os.path.exists(classes_txt_path):
        try:
            with open(classes_txt_path, 'r') as f:
                classes = [line.strip() for line in f.readlines() if line.strip()]
                class_names = {i: cls for i, cls in enumerate(classes)}
        except Exception as e:
            print(f"Warning: Could not load classes from classes.txt: {e}")
    
    # Fallback to notes.json
    if not class_names and os.path.exists(notes_json_path):
        try:
            with open(notes_json_path, 'r') as f:
                notes = json.load(f)
                if 'categories' in notes:
                    class_names = {cat['id']: cat['name'] for cat in notes['categories']}
        except Exception as e:
            print(f"Warning: Could not load classes from notes.json: {e}")
    
    return class_names

def parse_class_filter(class_filter: str, class_names: Dict[int, str]) -> Set[int]:
    """Parse class filter string and return set of class IDs"""
    if not class_filter:
        return set(class_names.keys())
    
    # Split by comma and strip whitespace
    filter_items = [item.strip() for item in class_filter.split(',')]
    class_ids = set()
    
    # Create reverse mapping for name lookups
    name_to_id = {name.lower(): id for id, name in class_names.items()}
    
    for item in filter_items:
        # Try as class ID first
        try:
            class_id = int(item)
            if class_id in class_names:
                class_ids.add(class_id)
            else:
                print(f"Warning: Class ID {class_id} not found in dataset")
        except ValueError:
            # Try as class name
            item_lower = item.lower()
            if item_lower in name_to_id:
                class_ids.add(name_to_id[item_lower])
            else:
                print(f"Warning: Class name '{item}' not found in dataset")
    
    return class_ids

def polygon_to_bbox(polygon_points: List[float]) -> Tuple[float, float, float, float] | None:
    """
    Convert polygon points to bounding box in YOLO format using Ultralytics segments2boxes utility
    
    Args:
        polygon_points: List of normalized coordinates [x1, y1, x2, y2, ..., xn, yn]
    Returns:
        Tuple of (center_x, center_y, width, height) in normalized coordinates
    """
    if len(polygon_points) < 6:  # Need at least 3 points (6 coordinates)
        return None
    
    if not ULTRALYTICS_AVAILABLE or segments2boxes is None:
        raise ImportError("Ultralytics is required for polygon-to-bbox conversion. Please install ultralytics: pip install ultralytics")
    
    try:
        # Reshape polygon points to (n, 2) format expected by segments2boxes
        points = np.array(polygon_points).reshape(-1, 2)
        # segments2boxes expects shape (1, n, 2) for batch processing
        segments = points[np.newaxis, ...]
        
        # Convert segments to bounding boxes
        boxes = segments2boxes(segments)
        
        # Extract the single box (x_center, y_center, width, height)
        if len(boxes) > 0:
            x_center, y_center, width, height = boxes[0]
            return float(x_center), float(y_center), float(width), float(height)
        else:
            return None
            
    except Exception as e:
        raise RuntimeError(f"Failed to convert polygon to bbox using Ultralytics segments2boxes: {e}")

def filter_label_file(label_path: str, allowed_classes: Set[int], convert_to_bbox: bool = False) -> Tuple[bool, List[str]]:
    """
    Filter a YOLO label file to only include allowed classes
    Args:
        label_path: Path to the label file
        allowed_classes: Set of allowed class IDs
        convert_to_bbox: If True, convert polygon/segmentation labels to bounding boxes
    Returns:
        (has_valid_annotations, filtered_lines)
    """
    if not os.path.exists(label_path):
        return False, []
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        filtered_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            if len(parts) < 2:  # Need at least class_id and one coordinate
                continue
            
            try:
                class_id = int(parts[0])
                if class_id not in allowed_classes:
                    continue
                
                # Check if this is a bounding box (5 parts) or polygon/segmentation (>5 parts)
                if len(parts) == 5:
                    # Standard YOLO bounding box format: class_id x y w h
                    filtered_lines.append(line + '\n')
                elif len(parts) > 5 and convert_to_bbox:
                    # Polygon/segmentation format: class_id x1 y1 x2 y2 ... xn yn
                    # Convert to bounding box
                    try:
                        polygon_points = [float(p) for p in parts[1:]]
                        bbox = polygon_to_bbox(polygon_points)
                        if bbox:
                            center_x, center_y, width, height = bbox
                            bbox_line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                            filtered_lines.append(bbox_line)
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert polygon to bbox in {label_path}: {line}")
                        continue
                elif len(parts) > 5 and not convert_to_bbox:
                    # Keep polygon/segmentation format as-is
                    filtered_lines.append(line + '\n')
                else:
                    # Invalid format
                    continue
                    
            except ValueError:
                continue
        
        return len(filtered_lines) > 0, filtered_lines
    
    except Exception as e:
        print(f"Warning: Error processing label file {label_path}: {e}")
        return False, []

def create_subset_split(source_split_dir: str, target_split_dir: str, 
                       allowed_classes: Set[int], class_names: Dict[int, str], 
                       convert_to_bbox: bool = False) -> Dict[str, int]:
    """Create a subset for one split (train/val/test)"""
    images_dir = os.path.join(source_split_dir, 'images')
    labels_dir = os.path.join(source_split_dir, 'labels')
    
    target_images_dir = os.path.join(target_split_dir, 'images')
    target_labels_dir = os.path.join(target_split_dir, 'labels')
    
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)
    
    if not os.path.exists(images_dir):
        return {'total_images': 0, 'kept_images': 0, 'kept_annotations': 0}
    
    stats = {'total_images': 0, 'kept_images': 0, 'kept_annotations': 0}
    
    # Process all images in the split
    for image_file in os.listdir(images_dir):
        if not image_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')):
            continue
        
        stats['total_images'] += 1
        image_path = os.path.join(images_dir, image_file)
        
        # Find corresponding label file
        image_stem = Path(image_file).stem
        label_file = image_stem + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # Filter the label file
        has_valid_annotations, filtered_lines = filter_label_file(label_path, allowed_classes, convert_to_bbox)
        
        if has_valid_annotations:
            # Copy image
            target_image_path = os.path.join(target_images_dir, image_file)
            shutil.copy2(image_path, target_image_path)
            
            # Save filtered labels
            target_label_path = os.path.join(target_labels_dir, label_file)
            with open(target_label_path, 'w') as f:
                f.writelines(filtered_lines)
            
            stats['kept_images'] += 1
            stats['kept_annotations'] += len(filtered_lines)
    
    return stats

def remap_class_ids(dataset_dir: str, allowed_classes: Set[int], class_names: Dict[int, str]) -> Dict[int, int]:
    """Create mapping from old class IDs to new sequential class IDs"""
    sorted_classes = sorted(allowed_classes)
    old_to_new = {old_id: new_id for new_id, old_id in enumerate(sorted_classes)}
    
    # Update all label files with new class IDs
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        if not os.path.exists(labels_dir):
            continue
        
        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
            
            label_path = os.path.join(labels_dir, label_file)
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                updated_lines = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        old_class_id = int(parts[0])
                        if old_class_id in old_to_new:
                            parts[0] = str(old_to_new[old_class_id])
                            updated_lines.append(' '.join(parts) + '\n')
                
                with open(label_path, 'w') as f:
                    f.writelines(updated_lines)
            
            except Exception as e:
                print(f"Warning: Error updating label file {label_path}: {e}")
    
    return old_to_new

def update_notes_json(source_file: str, target_file: str, allowed_classes: Set[int], 
                     class_names: Dict[int, str], class_id_mapping: Dict[int, int]):
    """Update notes.json file to only include filtered classes"""
    try:
        with open(source_file, 'r') as f:
            notes = json.load(f)
        
        filtered_categories = []
        
        # Filter categories to only include allowed classes
        if 'categories' in notes:
            for category in notes['categories']:
                if category['id'] in allowed_classes:
                    # Create new category with potentially remapped ID
                    new_category = category.copy()
                    if class_id_mapping:
                        new_category['id'] = class_id_mapping[category['id']]
                    filtered_categories.append(new_category)
            
            # Sort by new ID
            filtered_categories.sort(key=lambda x: x['id'])
            notes['categories'] = filtered_categories
            
            # Update info section if it exists
            if 'info' in notes:
                notes['info'] = notes['info'].copy()
                if 'description' in notes['info']:
                    notes['info']['description'] = f"Subset of original dataset with {len(filtered_categories)} classes"
                else:
                    notes['info']['description'] = f"Dataset subset with {len(filtered_categories)} classes"
        
        # Save updated notes.json
        with open(target_file, 'w') as f:
            json.dump(notes, f, indent=2)
        
        print(f"Updated notes.json with {len(filtered_categories)} classes")
        
    except Exception as e:
        print(f"Warning: Could not update notes.json: {e}")
        # Fallback: copy original file
        shutil.copy2(source_file, target_file)

def create_dataset_subset(source_dataset: str, target_dataset: str, class_filter: str, 
                         remap_classes: bool = True, convert_to_bbox: bool = False) -> str:
    """
    Create a subset of a dataset containing only specified classes
    
    Args:
        source_dataset: Path to source dataset directory
        target_dataset: Path to target dataset directory  
        class_filter: Comma-separated list of class names or IDs to include
        remap_classes: Whether to remap class IDs to be sequential starting from 0
        convert_to_bbox: Whether to convert polygon/segmentation labels to bounding boxes
    
    Returns:
        Path to created subset dataset
    """
    if not os.path.exists(source_dataset):
        raise ValueError(f"Source dataset not found: {source_dataset}")
    
    # Load class information
    class_names = load_class_names(source_dataset)
    if not class_names:
        raise ValueError(f"Could not load class information from {source_dataset}")
    
    print(f"📋 Available classes in source dataset:")
    for class_id, class_name in sorted(class_names.items()):
        print(f"  {class_id}: {class_name}")
    
    # Parse class filter
    allowed_classes = parse_class_filter(class_filter, class_names)
    if not allowed_classes:
        raise ValueError("No valid classes specified in filter")
    
    print(f"\n🎯 Creating subset with classes:")
    for class_id in sorted(allowed_classes):
        print(f"  {class_id}: {class_names[class_id]}")
    
    if convert_to_bbox:
        if not ULTRALYTICS_AVAILABLE or segments2boxes is None:
            raise ImportError("Ultralytics is required for polygon-to-bbox conversion. Please install ultralytics: pip install ultralytics")
        print(f"\n🔄 Converting polygon/segmentation labels to bounding boxes using Ultralytics segments2boxes")
    
    # Create target directory
    os.makedirs(target_dataset, exist_ok=True)
    
    # Process each split
    total_stats = {'total_images': 0, 'kept_images': 0, 'kept_annotations': 0}
    
    for split in ['train', 'val', 'test']:
        source_split_dir = os.path.join(source_dataset, split)
        target_split_dir = os.path.join(target_dataset, split)
        
        if os.path.exists(source_split_dir):
            print(f"\n📂 Processing {split} split...")
            split_stats = create_subset_split(source_split_dir, target_split_dir, 
                                            allowed_classes, class_names, convert_to_bbox)
            
            print(f"  Images: {split_stats['kept_images']}/{split_stats['total_images']} kept")
            print(f"  Annotations: {split_stats['kept_annotations']}")
            
            for key in total_stats:
                total_stats[key] += split_stats[key]
    
    # Remap class IDs if requested
    class_id_mapping = {}
    if remap_classes:
        print(f"\n🔄 Remapping class IDs to be sequential...")
        class_id_mapping = remap_class_ids(target_dataset, allowed_classes, class_names)
        print(f"  Class ID mapping: {class_id_mapping}")
    
    # Create new class names list
    if remap_classes:
        new_class_names = [class_names[old_id] for old_id in sorted(allowed_classes)]
    else:
        new_class_names = [class_names.get(i, f"class_{i}") for i in range(max(allowed_classes) + 1)]
    
    # Create configuration files
    config = {
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(allowed_classes),
        'names': new_class_names
    }
    
    # Save data.yaml
    data_yaml_path = os.path.join(target_dataset, 'data.yaml')
    save_yaml(config, data_yaml_path)
    
    # Save classes.txt
    classes_txt_path = os.path.join(target_dataset, 'classes.txt')
    with open(classes_txt_path, 'w') as f:
        for class_name in new_class_names:
            f.write(f"{class_name}\n")
    
    # Copy and update other files if they exist
    for filename in ['notes.json', 'README.md']:
        source_file = os.path.join(source_dataset, filename)
        if os.path.exists(source_file):
            target_file = os.path.join(target_dataset, filename)
            
            if filename == 'notes.json':
                # Update notes.json to only include filtered classes
                update_notes_json(source_file, target_file, allowed_classes, 
                                class_names, class_id_mapping if remap_classes else {})
            else:
                shutil.copy2(source_file, target_file)
    
    # Create subset info file
    subset_info = {
        'source_dataset': source_dataset,
        'target_dataset': target_dataset,
        'class_filter': class_filter,
        'allowed_classes': list(sorted(allowed_classes)),
        'class_names': {class_id: class_names[class_id] for class_id in allowed_classes},
        'remap_classes': remap_classes,
        'convert_to_bbox': convert_to_bbox,
        'class_id_mapping': class_id_mapping if remap_classes else {},
        'statistics': total_stats
    }
    
    subset_info_path = os.path.join(target_dataset, 'subset_info.json')
    with open(subset_info_path, 'w') as f:
        json.dump(subset_info, f, indent=2)
    
    print(f"\n✅ Subset dataset created successfully!")
    print(f"📊 Overall statistics:")
    print(f"  Total images processed: {total_stats['total_images']}")
    print(f"  Images kept: {total_stats['kept_images']}")
    print(f"  Total annotations: {total_stats['kept_annotations']}")
    print(f"  Classes: {len(allowed_classes)}")
    print(f"📁 Output directory: {target_dataset}")
    
    # Validate the created dataset
    try:
        stats = validate_dataset_structure(target_dataset)
        print(f"\n📋 Dataset validation:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"Warning: Could not validate dataset: {e}")
    
    return target_dataset

def main():
    parser = argparse.ArgumentParser(description='Create a subset of a dataset with specific classes')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to source dataset directory')
    parser.add_argument('--target', type=str, required=True,
                       help='Path to target subset dataset directory')
    parser.add_argument('--classes', type=str, required=True,
                       help='Comma-separated list of class names or IDs to include (e.g., "Red_Fruit,Green_Fruit" or "0,1,2")')
    parser.add_argument('--no-remap', action='store_true',
                       help='Do not remap class IDs to be sequential (keep original IDs)')
    parser.add_argument('--convert-to-bbox', action='store_true',
                       help='Convert polygon/segmentation labels to bounding boxes (requires ultralytics)')
    parser.add_argument('--list-classes', action='store_true',
                       help='List available classes in the source dataset and exit')
    
    args = parser.parse_args()
    
    # List classes mode
    if args.list_classes:
        try:
            class_names = load_class_names(args.source)
            if class_names:
                print(f"Available classes in {args.source}:")
                for class_id, class_name in sorted(class_names.items()):
                    print(f"  {class_id}: {class_name}")
            else:
                print(f"No classes found in {args.source}")
        except Exception as e:
            print(f"Error loading classes: {e}")
            return 1
        return 0
    
    try:
        create_dataset_subset(
            source_dataset=args.source,
            target_dataset=args.target,
            class_filter=args.classes,
            remap_classes=not args.no_remap,
            convert_to_bbox=args.convert_to_bbox
        )
        print(f"\n🎉 Dataset subset '{args.target}' created successfully!")
        
    except Exception as e:
        print(f"❌ Error creating dataset subset: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
