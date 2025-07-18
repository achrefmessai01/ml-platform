#!/usr/bin/env python3
"""
Extract Crops for Classification

This script extracts cropped images from object detection datasets to create
classification datasets. It reads YOLO format annotations and crops the images
according to the bounding boxes, saving them organized by class.

Features:
- Supports YOLO format datasets
- Configurable output directory structure
- Handles multiple classes
- Optional image resizing and padding
- Progress tracking
- Detailed logging

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
        self.padding = config.get('padding', 0.1)  # Extra padding around bbox (0.1 = 10%)
        self.min_crop_size = config.get('min_crop_size', 32)  # Minimum crop size
        self.save_format = config.get('save_format', 'JPEG')
        self.quality = config.get('quality', 95)
        
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
            'skipped_crops': 0
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
    
    def _apply_padding(self, x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Apply padding to bounding box"""
        width = x2 - x1
        height = y2 - y1
        
        pad_x = int(width * self.padding)
        pad_y = int(height * self.padding)
        
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
                class_id, x1, y1, x2, y2 = self._parse_yolo_annotation(annotation, img_width, img_height)
                
                # Get class name
                class_name = self.class_names[class_id]
                
                # Skip if not in include_labels (when include_labels is specified)
                if self.include_labels and class_name not in self.include_labels:
                    continue
                
                # Apply padding
                x1, y1, x2, y2 = self._apply_padding(x1, y1, x2, y2, img_width, img_height)
                
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
                
                # Resize if needed
                crop = self._resize_crop(crop)
                
                # Determine class name and output path
                if self.merge_labels:
                    output_class_name = self.merged_label_name
                else:
                    output_class_name = class_name
                
                # Generate output filename
                output_filename = f"{image_name}_crop_{crop_count:03d}.jpg"
                
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

def create_config_template():
    """Create a configuration template"""
    config_template = {
        'dataset_path': 'data/processed/farm_dataset',
        'output_dir': 'data/classification/big_dataset',
        'crop_size': None,  # [width, height] or null for original size
        'padding': 0.1,  # Extra padding around bbox (0.1 = 10%)
        'min_crop_size': 32,  # Minimum crop size in pixels
        'save_format': 'JPEG',  # JPEG or PNG
        'quality': 95,  # JPEG quality (1-100)
        'preserve_splits': True,  # Preserve train/val/test splits
        'merge_labels': False,  # Merge all labels into one class
        'merged_label_name': 'strawberry',  # Name for merged class (when merge_labels=True)
        'include_labels': None  # List of labels to include (null = all labels)
    }
    
    config_path = 'crop_extraction_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False)
    
    print(f"Configuration template created: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description='Extract crops for classification from object detection dataset')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--dataset_path', type=str, help='Path to YOLO dataset')
    parser.add_argument('--output_dir', type=str, help='Output directory for classification dataset')
    parser.add_argument('--crop_size', type=str, help='Target crop size as WIDTHxHEIGHT (e.g., 224x224)')
    parser.add_argument('--padding', type=float, default=0.1, help='Extra padding around bbox (default: 0.1)')
    parser.add_argument('--min_crop_size', type=int, default=32, help='Minimum crop size (default: 32)')
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
