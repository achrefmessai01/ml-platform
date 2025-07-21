#!/usr/bin/env python3
"""
Extract Crops for Classification

This script extracts cropped images from object detection datasets to create
classification datasets. It reads YOLO format annotations and crops the images
according to the bounding boxes, saving them organized by class.

Features:
- Supports YOLO format datasets (AABB and OBB)
- Auto-detects annotation format (5 values = AABB, 8+ values = OBB)
- Configurable output directory structure
- Handles multiple classes
- Optional image resizing and padding
- Progress tracking
- Detailed logging
- Oriented bounding box support with perspective transformation

Supported Formats:
- AABB: class_id x_center y_center width height
- OBB: class_id x1 y1 x2 y2 x3 y3 x4 y4

Usage:
    python scripts/extract_crops_for_classification.py --config config.yaml
    python scripts/extract_crops_for_classification.py --dataset_path data/processed/farm_dataset --output_dir data/classification/big_dataset
"""

import os
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional
import json
import shutil

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClassificationCropExtractor:
    """Extract crops from object detection datasets for classification training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_path = Path(config['dataset_path'])
        self.output_dir = Path(config['output_dir'])
        self.class_names = self._load_class_names()
        
        # Crop settings
        self.crop_size = config.get('crop_size', None)  # (width, height) or None for original
        # Handle string "None" or "null" values in YAML
        if isinstance(self.crop_size, str) and self.crop_size.lower() in ['none', 'null']:
            self.crop_size = None
        self.padding = config.get('padding', 0.1)  # Default padding around bbox (0.1 = 10%)
        self.class_padding = config.get('class_padding', {})  # Class-specific padding overrides
        self.min_crop_size = config.get('min_crop_size', 0)  # Minimum crop size
        self.save_format = config.get('save_format', 'JPEG')
        self.quality = config.get('quality', 100)
        
        # Split and label settings
        self.preserve_splits = config.get('preserve_splits', True)  # Preserve train/val/test splits
        self.merge_labels = config.get('merge_labels', False)  # Merge all labels into one class
        self.merged_label_name = config.get('merged_label_name', 'strawberry')  # Name for merged class
        self.include_labels = config.get('include_labels', None)  # List of labels to include (None = all labels)
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_crops': 0,
            'crops_per_class': {},
            'crops_per_split': {'train': 0, 'val': 0, 'test': 0} if self.preserve_splits else {},
            'skipped_crops': 0,
            'aabb_count': 0,  # Count of axis-aligned bounding boxes
            'obb_count': 0,   # Count of oriented bounding boxes
            'annotation_format': None  # Will be detected: 'AABB', 'OBB', or 'MIXED'
        }
        
        # Create output directories
        self._create_output_dirs()
    
    def _load_class_names(self) -> List[str]:
        """Load class names from dataset"""
        # Try data.yaml first
        data_yaml_path = self.dataset_path / 'data.yaml'
        if data_yaml_path.exists():
            with open(data_yaml_path, 'r') as f:
                data = yaml.safe_load(f)
                if 'names' in data:
                    if isinstance(data['names'], dict):
                        return [data['names'][i] for i in sorted(data['names'].keys())]
                    else:
                        return data['names']
        
        # Try classes.txt
        classes_path = self.dataset_path / 'classes.txt'
        if classes_path.exists():
            with open(classes_path, 'r') as f:
                return [line.strip() for line in f.readlines()]
        
        # Fallback: scan label files to find max class index
        logger.warning("No class names found, will use generic names")
        max_class = 0
        for split in ['train', 'val', 'test']:
            labels_dir = self.dataset_path / split / 'labels'
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                max_class = max(max_class, class_id)
        
        return [f"class_{i}" for i in range(max_class + 1)]
    
    def _create_output_dirs(self):
        """Create output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine class names to create directories for
        if self.merge_labels:
            class_names = [self.merged_label_name]
        else:
            # Filter class names based on include_labels if specified
            if self.include_labels:
                class_names = [name for name in self.class_names if name in self.include_labels]
                if not class_names:
                    logger.warning(f"No matching classes found for include_labels: {self.include_labels}")
                    logger.warning(f"Available classes: {self.class_names}")
                    class_names = []
            else:
                class_names = self.class_names
        
        # Create split directories if preserving splits
        if self.preserve_splits:
            for split in ['train', 'val', 'test']:
                split_dir = self.output_dir / split
                split_dir.mkdir(exist_ok=True)
                
                for class_name in class_names:
                    class_dir = split_dir / class_name
                    class_dir.mkdir(exist_ok=True)
        else:
            # Create class directories directly under output_dir
            for class_name in class_names:
                class_dir = self.output_dir / class_name
                class_dir.mkdir(exist_ok=True)
        
        # Initialize statistics
        for class_name in class_names:
            self.stats['crops_per_class'][class_name] = 0
    
    def _parse_yolo_annotation(self, annotation_line: str, img_width: int, img_height: int) -> Tuple[int, int, int, int, int]:
        """Parse YOLO format annotation line"""
        parts = annotation_line.strip().split()
        class_id = int(parts[0])
        x_center = float(parts[1]) * img_width
        y_center = float(parts[2]) * img_height
        width = float(parts[3]) * img_width
        height = float(parts[4]) * img_height
        
        # Convert to bbox coordinates
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        
        return class_id, x1, y1, x2, y2
    
    def _parse_obb_annotation(self, annotation_line: str, img_width: int, img_height: int) -> Tuple[int, np.ndarray]:
        """Parse oriented bounding box annotation line"""
        parts = annotation_line.strip().split()
        class_id = int(parts[0])
        
        # Parse 4 corner points (8 coordinates)
        points = []
        for i in range(1, 9, 2):
            x = float(parts[i]) * img_width
            y = float(parts[i + 1]) * img_height
            points.append([x, y])
        
        # Convert to numpy array
        points = np.array(points, dtype=np.float32)
        
        return class_id, points
    
    def _detect_annotation_format(self, annotation_line: str) -> str:
        """Detect if annotation is AABB (5 values) or OBB (9 values)"""
        parts = annotation_line.strip().split()
        if len(parts) == 5:
            return 'AABB'
        elif len(parts) == 9:
            return 'OBB'
        else:
            raise ValueError(f"Invalid annotation format: expected 5 (AABB) or 9 (OBB) values, got {len(parts)}")
    
    def _extract_obb_crop(self, image: np.ndarray, points: np.ndarray, class_name: str) -> np.ndarray:
        """Extract crop from oriented bounding box using perspective transformation"""
        # Get padding for this specific class
        padding = self.class_padding.get(class_name, self.padding)
        
        # Calculate bounding rectangle of the oriented box
        x_min = np.min(points[:, 0])
        x_max = np.max(points[:, 0])
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        
        # Apply padding to bounding rectangle
        width = x_max - x_min
        height = y_max - y_min
        pad_x = width * padding
        pad_y = height * padding
        
        # Expand points outward from center
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Apply padding by scaling points from center
        scale_x = (width + 2 * pad_x) / width
        scale_y = (height + 2 * pad_y) / height
        
        padded_points = points.copy()
        for i in range(4):
            # Scale from center
            padded_points[i, 0] = center_x + (points[i, 0] - center_x) * scale_x
            padded_points[i, 1] = center_y + (points[i, 1] - center_y) * scale_y
        
        # Ensure points are within image bounds
        img_height, img_width = image.shape[:2]
        padded_points[:, 0] = np.clip(padded_points[:, 0], 0, img_width - 1)
        padded_points[:, 1] = np.clip(padded_points[:, 1], 0, img_height - 1)
        
        # Calculate dimensions of the oriented bounding box
        # Use the original points (before padding) to maintain aspect ratio
        edge1 = np.linalg.norm(points[1] - points[0])
        edge2 = np.linalg.norm(points[2] - points[1])
        
        # Determine output dimensions (add padding)
        out_width = int(edge1 + 2 * pad_x)
        out_height = int(edge2 + 2 * pad_y)
        
        # Define destination rectangle (with padding included)
        dst_points = np.array([
            [0, 0],
            [out_width - 1, 0],
            [out_width - 1, out_height - 1],
            [0, out_height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        try:
            transform_matrix = cv2.getPerspectiveTransform(padded_points, dst_points)
            
            # Apply perspective transformation
            crop = cv2.warpPerspective(image, transform_matrix, (out_width, out_height))
            
            return crop
        except cv2.error as e:
            logger.warning(f"Failed to apply perspective transformation: {e}")
            # Fallback: extract axis-aligned bounding box
            x1, y1 = int(np.min(padded_points[:, 0])), int(np.min(padded_points[:, 1]))
            x2, y2 = int(np.max(padded_points[:, 0])), int(np.max(padded_points[:, 1]))
            return image[y1:y2, x1:x2]
    
    def _apply_padding(self, x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int, class_name: str) -> Tuple[int, int, int, int]:
        """Apply padding to bounding box (class-specific or default)"""
        # Get padding for this specific class, fall back to default padding
        padding = self.class_padding.get(class_name, self.padding)
        
        width = x2 - x1
        height = y2 - y1
        
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_width, x2 + pad_x)
        y2 = min(img_height, y2 + pad_y)
        
        return x1, y1, x2, y2
    
    def _resize_crop(self, crop: np.ndarray) -> np.ndarray:
        """Resize crop if crop_size is specified"""
        if self.crop_size is None:
            return crop
        
        target_width, target_height = self.crop_size
        current_height, current_width = crop.shape[:2]
        
        # Calculate scaling to fit within target size while maintaining aspect ratio
        scale = min(target_width / current_width, target_height / current_height)
        
        if scale < 1.0:  # Only downscale, never upscale
            new_width = int(current_width * scale)
            new_height = int(current_height * scale)
            crop = cv2.resize(crop, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Pad to exact target size if needed
        if crop.shape[:2] != (target_height, target_width):
            # Create a black canvas
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
            # Center the resized crop
            y_offset = (target_height - crop.shape[0]) // 2
            x_offset = (target_width - crop.shape[1]) // 2
            
            canvas[y_offset:y_offset + crop.shape[0], x_offset:x_offset + crop.shape[1]] = crop
            crop = canvas
        
        return crop
    
    def _extract_crops_from_image(self, image_path: Path, label_path: Path, split: Optional[str] = None):
        """Extract all crops from a single image"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return
        
        img_height, img_width = image.shape[:2]
        
        # Load annotations
        if not label_path.exists():
            logger.debug(f"No label file found for {image_path}")
            return
        
        with open(label_path, 'r') as f:
            annotations = f.readlines()
        
        if not annotations:
            return
        
        image_name = image_path.stem
        crop_count = 0
        
        for annotation in annotations:
            annotation = annotation.strip()
            if not annotation:
                continue
            
            try:
                # Detect annotation format
                format_type = self._detect_annotation_format(annotation)
                
                # Update format statistics
                if format_type == 'AABB':
                    self.stats['aabb_count'] += 1
                elif format_type == 'OBB':
                    self.stats['obb_count'] += 1
                
                # Set overall format detection
                if self.stats['annotation_format'] is None:
                    self.stats['annotation_format'] = format_type
                elif self.stats['annotation_format'] != format_type:
                    self.stats['annotation_format'] = 'MIXED'
                
                # Initialize variables
                crop = None
                class_name = None
                
                # Parse annotation based on format
                if format_type == 'AABB':
                    class_id, x1, y1, x2, y2 = self._parse_yolo_annotation(annotation, img_width, img_height)
                    
                    # Get class name
                    class_name = self.class_names[class_id]
                    
                    # Skip if not in include_labels (when include_labels is specified)
                    if self.include_labels and class_name not in self.include_labels:
                        continue
                    
                    # Apply padding (class-specific)
                    x1, y1, x2, y2 = self._apply_padding(x1, y1, x2, y2, img_width, img_height, class_name)
                    
                    # Check minimum crop size
                    crop_width = x2 - x1
                    crop_height = y2 - y1
                    if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
                        self.stats['skipped_crops'] += 1
                        continue
                    
                    # Extract crop
                    crop = image[y1:y2, x1:x2]
                    if crop.size == 0:
                        self.stats['skipped_crops'] += 1
                        continue
                
                elif format_type == 'OBB':
                    class_id, points = self._parse_obb_annotation(annotation, img_width, img_height)
                    
                    # Get class name
                    class_name = self.class_names[class_id]
                    
                    # Skip if not in include_labels (when include_labels is specified)
                    if self.include_labels and class_name not in self.include_labels:
                        continue
                    
                    # Extract oriented crop with padding
                    crop = self._extract_obb_crop(image, points, class_name)
                    if crop is None or crop.size == 0:
                        self.stats['skipped_crops'] += 1
                        continue
                    
                    # Check minimum crop size
                    crop_height, crop_width = crop.shape[:2]
                    if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
                        self.stats['skipped_crops'] += 1
                        continue
                
                # Skip if crop or class_name wasn't set (shouldn't happen)
                if crop is None or class_name is None:
                    self.stats['skipped_crops'] += 1
                    continue
                
                # Resize if needed
                crop = self._resize_crop(crop)
                
                # Determine class name and output path
                if self.merge_labels:
                    output_class_name = self.merged_label_name
                else:
                    output_class_name = class_name
                
                # Generate output filename with format indicator
                format_suffix = 'aabb' if format_type == 'AABB' else 'obb'
                output_filename = f"{image_name}_crop_{crop_count:03d}_{format_suffix}.jpg"
                
                # Determine output directory based on split preservation
                if self.preserve_splits and split:
                    output_path = self.output_dir / split / output_class_name / output_filename
                else:
                    output_path = self.output_dir / output_class_name / output_filename
                
                # Save crop
                if self.save_format.upper() == 'JPEG':
                    cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                else:
                    cv2.imwrite(str(output_path), crop)
                
                # Update statistics
                self.stats['crops_per_class'][output_class_name] = self.stats['crops_per_class'].get(output_class_name, 0) + 1
                self.stats['total_crops'] += 1
                if self.preserve_splits and split:
                    self.stats['crops_per_split'][split] += 1
                crop_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing annotation in {label_path}: {e}")
                self.stats['skipped_crops'] += 1
    
    def extract_crops(self):
        """Extract crops from all images in the dataset"""
        logger.info(f"Starting crop extraction from {self.dataset_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Preserve splits: {self.preserve_splits}")
        logger.info(f"Merge labels: {self.merge_labels}")
        logger.info(f"Default padding: {self.padding}")
        if self.class_padding:
            logger.info("Class-specific padding:")
            for class_name, padding in self.class_padding.items():
                logger.info(f"  {class_name}: {padding}")
        if self.merge_labels:
            logger.info(f"Merged label name: {self.merged_label_name}")
        if self.include_labels:
            logger.info(f"Include labels: {self.include_labels}")
            logger.info(f"Available classes: {self.class_names}")
        else:
            logger.info(f"Classes: {self.class_names}")
        
        # Process all splits
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        all_images = []
        
        for split in ['train', 'val', 'test']:
            images_dir = self.dataset_path / split / 'images'
            labels_dir = self.dataset_path / split / 'labels'
            
            if images_dir.exists():
                for img_path in images_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        label_path = labels_dir / f"{img_path.stem}.txt"
                        all_images.append((img_path, label_path, split))
        
        if not all_images:
            logger.error("No images found in dataset")
            return
        
        logger.info(f"Found {len(all_images)} images to process")
        
        # Process images with progress bar
        for img_path, label_path, split in tqdm(all_images, desc="Extracting crops"):
            self._extract_crops_from_image(img_path, label_path, split)
            self.stats['total_images'] += 1
        
        self._save_statistics()
        self._print_summary()
    
    def _save_statistics(self):
        """Save extraction statistics"""
        stats_path = self.output_dir / 'extraction_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def _print_summary(self):
        """Print extraction summary"""
        logger.info("=" * 50)
        logger.info("CROP EXTRACTION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total images processed: {self.stats['total_images']}")
        logger.info(f"Total crops extracted: {self.stats['total_crops']}")
        logger.info(f"Skipped crops: {self.stats['skipped_crops']}")
        
        # Annotation format statistics
        logger.info(f"\nAnnotation format detected: {self.stats['annotation_format']}")
        if self.stats['aabb_count'] > 0:
            logger.info(f"Axis-aligned bounding boxes (AABB): {self.stats['aabb_count']}")
        if self.stats['obb_count'] > 0:
            logger.info(f"Oriented bounding boxes (OBB): {self.stats['obb_count']}")
        
        if self.preserve_splits:
            logger.info("\nCrops per split:")
            for split, count in self.stats['crops_per_split'].items():
                logger.info(f"  {split}: {count}")
        
        logger.info("\nCrops per class:")
        for class_name, count in self.stats['crops_per_class'].items():
            logger.info(f"  {class_name}: {count}")
        
        logger.info(f"\nOutput directory: {self.output_dir}")
        
        if self.preserve_splits:
            logger.info("\nDirectory structure:")
            logger.info(f"  {self.output_dir}/")
            for split in ['train', 'val', 'test']:
                if self.stats['crops_per_split'][split] > 0:
                    logger.info(f"  ├── {split}/")
                    for class_name in self.stats['crops_per_class'].keys():
                        logger.info(f"  │   └── {class_name}/")
        else:
            logger.info("\nDirectory structure:")
            logger.info(f"  {self.output_dir}/")
            for class_name in self.stats['crops_per_class'].keys():
                logger.info(f"  └── {class_name}/")
        
        # Additional info for OBB support
        if self.stats['obb_count'] > 0:
            logger.info(f"\n💡 Oriented bounding box support:")
            logger.info(f"   • OBB crops extracted using perspective transformation")
            logger.info(f"   • Padding applied by scaling from object center")
            logger.info(f"   • Filenames include '_obb' suffix for identification")

def create_config_template():
    """Create a configuration template"""
    config_template = {
        'dataset_path': 'data/processed/farm_dataset',
        'output_dir': 'data/classification/big_dataset',
        'crop_size': None,  # [width, height] or null for original size
        'padding': 0.1,  # Default padding around bbox (0.1 = 10%)
        'class_padding': {},  # Class-specific padding overrides
        'min_crop_size': 0,  # Minimum crop size in pixels
        'save_format': 'JPEG',  # JPEG or PNG
        'quality': 100,  # JPEG quality (1-100)
        'preserve_splits': True,  # Preserve train/val/test splits
        'merge_labels': False,  # Merge all labels into one class
        'merged_label_name': 'strawberry',  # Name for merged class (when merge_labels=True)
        'include_labels': None  # List of labels to include (null = all labels)
    }
    
    config_path = 'crop_extraction_config.yaml'
    with open(config_path, 'w') as f:
        # Write the config with custom formatting to include examples
        f.write("# Crop Extraction Configuration\n")
        f.write("# This file configures the crop extraction script\n\n")
        
        f.write(f"dataset_path: '{config_template['dataset_path']}'\n")
        f.write(f"output_dir: '{config_template['output_dir']}'\n\n")
        
        f.write("# Crop settings\n")
        f.write(f"crop_size: null  # [width, height] or null for original size\n")
        f.write(f"padding: {config_template['padding']}  # Default padding around bbox (0.1 = 10%)\n")
        f.write("class_padding:  # Class-specific padding overrides\n")
        f.write("  # Flower: 0.2        # 20% padding for flowers\n")
        f.write("  # Red_Fruit: 0.05    # 5% padding for red fruits\n")
        f.write("  # Green_Fruit: 0.15  # 15% padding for green fruits\n")
        f.write(f"min_crop_size: {config_template['min_crop_size']}  # Minimum crop size in pixels\n\n")
        
        f.write("# Output settings\n")
        f.write(f"save_format: '{config_template['save_format']}'\n")
        f.write(f"quality: {config_template['quality']}  # JPEG quality (1-100)\n\n")
        
        f.write("# Dataset organization\n")
        f.write(f"preserve_splits: {str(config_template['preserve_splits']).lower()}\n")
        f.write(f"merge_labels: {str(config_template['merge_labels']).lower()}\n")
        f.write(f"merged_label_name: '{config_template['merged_label_name']}'\n")
        f.write(f"include_labels: {config_template['include_labels']}  # List of labels to include (null = all labels)\n")
    
    print(f"Configuration template created: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description='Extract crops for classification from object detection dataset (supports AABB and OBB)')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--dataset_path', type=str, help='Path to YOLO dataset')
    parser.add_argument('--output_dir', type=str, help='Output directory for classification dataset')
    parser.add_argument('--crop_size', type=str, help='Target crop size as WIDTHxHEIGHT (e.g., 224x224)')
    parser.add_argument('--padding', type=float, default=0.1, help='Default padding around bbox (default: 0.1)')
    parser.add_argument('--class_padding', type=str, nargs='+', help='Class-specific padding (e.g., --class_padding Flower=0.2 Red_Fruit=0.05)')
    parser.add_argument('--min_crop_size', type=int, default=0, help='Minimum crop size (default: 32)')
    parser.add_argument('--preserve_splits', action='store_true', default=True, help='Preserve train/val/test splits (default: True)')
    parser.add_argument('--no_preserve_splits', action='store_false', dest='preserve_splits', help='Do not preserve train/val/test splits')
    parser.add_argument('--merge_labels', action='store_true', help='Merge all labels into one class')
    parser.add_argument('--merged_label_name', type=str, default='strawberry', help='Name for merged class (default: strawberry)')
    parser.add_argument('--include_labels', type=str, nargs='+', help='List of labels to include (e.g., --include_labels Red_Fruit Green_Fruit)')
    parser.add_argument('--create_config', action='store_true', help='Create configuration template')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_config_template()
        return
    
    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
    
    # Override with command line arguments
    if args.dataset_path:
        config['dataset_path'] = args.dataset_path
    if args.output_dir:
        config['output_dir'] = args.output_dir
    if args.crop_size:
        width, height = map(int, args.crop_size.split('x'))
        config['crop_size'] = [width, height]
    if args.padding is not None:
        config['padding'] = args.padding
    if args.class_padding is not None:
        class_padding = {}
        for class_pad in args.class_padding:
            if '=' in class_pad:
                class_name, padding_str = class_pad.split('=', 1)
                try:
                    padding_value = float(padding_str)
                    class_padding[class_name] = padding_value
                except ValueError:
                    logger.warning(f"Invalid padding value for class {class_name}: {padding_str}")
            else:
                logger.warning(f"Invalid class padding format: {class_pad}. Expected format: ClassName=0.1")
        if class_padding:
            config['class_padding'] = class_padding
    if args.min_crop_size is not None:
        config['min_crop_size'] = args.min_crop_size
    if 'preserve_splits' not in config:
        config['preserve_splits'] = args.preserve_splits
    if args.merge_labels:
        config['merge_labels'] = True
    if args.merged_label_name and args.merged_label_name != 'strawberry':  # Only override if not default
        config['merged_label_name'] = args.merged_label_name
    if args.include_labels:
        config['include_labels'] = args.include_labels
    
    # Validate required parameters
    if 'dataset_path' not in config or 'output_dir' not in config:
        logger.error("dataset_path and output_dir must be specified")
        logger.info("Use --create_config to generate a configuration template")
        return
    
    # Extract crops
    extractor = ClassificationCropExtractor(config)
    extractor.extract_crops()

if __name__ == "__main__":
    main()
