#!/usr/bin/env python3
"""
Relabel Dataset with YOLO Classifier

This script takes a YOLO object detection dataset, crops the labeled objects,
runs inference on them using a YOLO classifier, and creates a new dataset
with the predicted labels replacing the original labels.

Features:
- Crops objects from images based on original labels
- Runs YOLO classifier inference on cropped regions
- Creates new dataset with predicted labels
- Preserves original dataset structure (train/val/test)
- Handles confidence thresholds and filtering
- Supports batch processing for efficiency

Usage:
    python scripts/relabel_dataset.py --config relabel_config.yaml
    python scripts/relabel_dataset.py --dataset_path data/processed/farm_dataset --classifier_model models/strawberry_classifier.pt --output_dir data/relabeled/farm_dataset
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
from ultralytics import YOLO

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetRelabeler:
    """Relabel dataset using YOLO classifier predictions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_path = Path(config['dataset_path'])
        self.output_dir = Path(config['output_dir'])
        self.classifier_model_path = config['classifier_model']
        
        # Load YOLO classifier
        logger.info(f"Loading YOLO classifier: {self.classifier_model_path}")
        self.classifier = YOLO(self.classifier_model_path)
        
        # Get class names from classifier
        self.classifier_classes = self.classifier.names
        logger.info(f"Classifier classes: {self.classifier_classes}")
        
        # Load original dataset class names
        self.original_classes = self._load_original_classes()
        logger.info(f"Original classes: {self.original_classes}")
        
        # Processing settings
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.crop_padding = config.get('crop_padding', 0.0)
        self.min_crop_size = config.get('min_crop_size', 0)
        self.preserve_splits = config.get('preserve_splits', True)
        self.skip_low_confidence = config.get('skip_low_confidence', True)
        
        # Statistics
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'relabeled_labels': 0,
            'skipped_labels': 0,
            'original_class_counts': {},
            'new_class_counts': {},
            'confidence_stats': []
        }
        
        # Create output directory structure
        self._create_output_structure()
    
    def _load_original_classes(self) -> List[str]:
        """Load original dataset class names"""
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
        logger.warning("No class names found, scanning label files...")
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
    
    def _create_output_structure(self):
        """Create output directory structure"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create split directories
        for split in ['train', 'val', 'test']:
            split_dir = self.output_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Create images and labels subdirectories
            (split_dir / 'images').mkdir(exist_ok=True)
            (split_dir / 'labels').mkdir(exist_ok=True)
        
        # Create new classes.txt with classifier classes
        classes_file = self.output_dir / 'classes.txt'
        with open(classes_file, 'w') as f:
            for class_name in self.classifier_classes.values():
                f.write(f"{class_name}\n")
        
        # Create new data.yaml
        data_yaml = self.output_dir / 'data.yaml'
        data_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': len(self.classifier_classes),
            'names': list(self.classifier_classes.values())
        }
        with open(data_yaml, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        
        logger.info(f"Created output structure at {self.output_dir}")
    
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
        
        pad_x = int(width * self.crop_padding)
        pad_y = int(height * self.crop_padding)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_width, x2 + pad_x)
        y2 = min(img_height, y2 + pad_y)
        
        return x1, y1, x2, y2
    
    def _classify_crop(self, crop: np.ndarray) -> Tuple[int, float]:
        """Classify a crop using YOLO classifier"""
        try:
            # Run inference
            results = self.classifier(crop, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'probs') and result.probs is not None:
                    # Get top prediction
                    top_class = result.probs.top1
                    confidence = result.probs.top1conf.item()
                    return top_class, confidence
            
            return -1, 0.0
            
        except Exception as e:
            logger.warning(f"Error during classification: {e}")
            return -1, 0.0
    
    def _convert_to_yolo_format(self, x1: int, y1: int, x2: int, y2: int, img_width: int, img_height: int) -> Tuple[float, float, float, float]:
        """Convert bbox coordinates to YOLO format"""
        x_center = (x1 + x2) / 2.0 / img_width
        y_center = (y1 + y2) / 2.0 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        return x_center, y_center, width, height
    
    def _process_image(self, image_path: Path, label_path: Path, output_image_path: Path, output_label_path: Path) -> Dict:
        """Process a single image and its labels"""
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning(f"Could not load image: {image_path}")
            return {'processed': False, 'error': 'Could not load image'}
        
        img_height, img_width = image.shape[:2]
        
        # Copy image to output directory
        shutil.copy2(image_path, output_image_path)
        
        # Load original annotations
        if not label_path.exists():
            logger.debug(f"No label file found for {image_path}")
            # Create empty label file
            output_label_path.touch()
            return {'processed': True, 'labels': 0, 'relabeled': 0}
        
        with open(label_path, 'r') as f:
            annotations = f.readlines()
        
        if not annotations:
            # Create empty label file
            output_label_path.touch()
            return {'processed': True, 'labels': 0, 'relabeled': 0}
        
        new_annotations = []
        labels_processed = 0
        labels_relabeled = 0
        
        for annotation in annotations:
            annotation = annotation.strip()
            if not annotation:
                continue
            
            try:
                original_class_id, x1, y1, x2, y2 = self._parse_yolo_annotation(annotation, img_width, img_height)
                
                # Update original class counts
                original_class_name = self.original_classes[original_class_id]
                self.stats['original_class_counts'][original_class_name] = self.stats['original_class_counts'].get(original_class_name, 0) + 1
                
                # Apply padding (disabled - no padding)
                # x1, y1, x2, y2 = self._apply_padding(x1, y1, x2, y2, img_width, img_height)
                
                # Check minimum crop size (disabled - no minimum size)
                # crop_width = x2 - x1
                # crop_height = y2 - y1
                # if crop_width < self.min_crop_size or crop_height < self.min_crop_size:
                #     logger.debug(f"Skipping small crop: {crop_width}x{crop_height}")
                #     self.stats['skipped_labels'] += 1
                #     continue
                
                # Extract crop
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    logger.debug(f"Empty crop, skipping")
                    self.stats['skipped_labels'] += 1
                    continue
                
                # Classify crop
                predicted_class, confidence = self._classify_crop(crop)
                
                # Check confidence threshold
                if confidence < self.confidence_threshold:
                    if self.skip_low_confidence:
                        logger.debug(f"Low confidence prediction ({confidence:.3f}), skipping")
                        self.stats['skipped_labels'] += 1
                        continue
                    else:
                        logger.debug(f"Low confidence prediction ({confidence:.3f}), keeping")
                
                # Update statistics
                self.stats['confidence_stats'].append(confidence)
                labels_processed += 1
                
                if predicted_class != -1:
                    # Convert back to YOLO format
                    x_center, y_center, width, height = self._convert_to_yolo_format(x1, y1, x2, y2, img_width, img_height)
                    
                    # Create new annotation
                    new_annotation = f"{predicted_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    new_annotations.append(new_annotation)
                    
                    # Update new class counts
                    new_class_name = self.classifier_classes[predicted_class]
                    self.stats['new_class_counts'][new_class_name] = self.stats['new_class_counts'].get(new_class_name, 0) + 1
                    
                    labels_relabeled += 1
                else:
                    logger.debug(f"Classification failed, skipping label")
                    self.stats['skipped_labels'] += 1
                
            except Exception as e:
                logger.warning(f"Error processing annotation in {label_path}: {e}")
                self.stats['skipped_labels'] += 1
        
        # Write new label file
        with open(output_label_path, 'w') as f:
            for annotation in new_annotations:
                f.write(annotation + '\n')
        
        return {
            'processed': True,
            'labels': labels_processed,
            'relabeled': labels_relabeled
        }
    
    def relabel_dataset(self):
        """Relabel the entire dataset"""
        logger.info(f"Starting dataset relabeling from {self.dataset_path}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Classifier model: {self.classifier_model_path}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        logger.info(f"Crop padding: {self.crop_padding}")
        logger.info(f"Skip low confidence: {self.skip_low_confidence}")
        
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
                        output_img_path = self.output_dir / split / 'images' / img_path.name
                        output_label_path = self.output_dir / split / 'labels' / f"{img_path.stem}.txt"
                        
                        all_images.append({
                            'image_path': img_path,
                            'label_path': label_path,
                            'output_image_path': output_img_path,
                            'output_label_path': output_label_path,
                            'split': split
                        })
        
        logger.info(f"Found {len(all_images)} images to process")
        
        # Process images with progress bar
        with tqdm(all_images, desc="Relabeling dataset") as pbar:
            for item in pbar:
                result = self._process_image(
                    item['image_path'],
                    item['label_path'],
                    item['output_image_path'],
                    item['output_label_path']
                )
                
                if result['processed']:
                    self.stats['total_images'] += 1
                    self.stats['total_labels'] += result['labels']
                    self.stats['relabeled_labels'] += result['relabeled']
                
                # Update progress bar
                pbar.set_postfix({
                    'Images': self.stats['total_images'],
                    'Labels': self.stats['relabeled_labels'],
                    'Skipped': self.stats['skipped_labels']
                })
        
        # Save statistics
        self._save_statistics()
        
        # Print summary
        self._print_summary()
    
    def _save_statistics(self):
        """Save processing statistics to file"""
        stats_file = self.output_dir / 'relabeling_stats.json'
        
        # Convert numpy types to Python types for JSON serialization
        stats_copy = self.stats.copy()
        if stats_copy['confidence_stats']:
            stats_copy['avg_confidence'] = float(np.mean(stats_copy['confidence_stats']))
            stats_copy['min_confidence'] = float(np.min(stats_copy['confidence_stats']))
            stats_copy['max_confidence'] = float(np.max(stats_copy['confidence_stats']))
        
        with open(stats_file, 'w') as f:
            json.dump(stats_copy, f, indent=2)
        
        logger.info(f"Statistics saved to {stats_file}")
    
    def _print_summary(self):
        """Print relabeling summary"""
        logger.info("=" * 50)
        logger.info("DATASET RELABELING SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total images processed: {self.stats['total_images']}")
        logger.info(f"Total labels processed: {self.stats['total_labels']}")
        logger.info(f"Labels relabeled: {self.stats['relabeled_labels']}")
        logger.info(f"Labels skipped: {self.stats['skipped_labels']}")
        
        if self.stats['confidence_stats']:
            logger.info(f"Average confidence: {np.mean(self.stats['confidence_stats']):.3f}")
            logger.info(f"Min confidence: {np.min(self.stats['confidence_stats']):.3f}")
            logger.info(f"Max confidence: {np.max(self.stats['confidence_stats']):.3f}")
        
        logger.info("\nOriginal class distribution:")
        for class_name, count in self.stats['original_class_counts'].items():
            logger.info(f"  {class_name}: {count}")
        
        logger.info("\nNew class distribution:")
        for class_name, count in self.stats['new_class_counts'].items():
            logger.info(f"  {class_name}: {count}")
        
        logger.info(f"\nOutput directory: {self.output_dir}")
        logger.info("\nDataset structure:")
        logger.info(f"  {self.output_dir}/")
        logger.info(f"  ├── train/")
        logger.info(f"  │   ├── images/")
        logger.info(f"  │   └── labels/")
        logger.info(f"  ├── val/")
        logger.info(f"  │   ├── images/")
        logger.info(f"  │   └── labels/")
        logger.info(f"  ├── test/")
        logger.info(f"  │   ├── images/")
        logger.info(f"  │   └── labels/")
        logger.info(f"  ├── data.yaml")
        logger.info(f"  ├── classes.txt")
        logger.info(f"  └── relabeling_stats.json")

def create_config_template():
    """Create a configuration template"""
    config_template = {
        'dataset_path': 'data/processed/farm_dataset',
        'output_dir': 'data/relabeled/farm_dataset',
        'classifier_model': 'models/strawberry_classifier.pt',
        'confidence_threshold': 0.5,
        'crop_padding': 0.0,
        'min_crop_size': 0,
        'preserve_splits': True,
        'skip_low_confidence': True
    }
    
    config_path = 'relabel_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False)
    
    print(f"Configuration template created: {config_path}")
    return config_path

def main():
    parser = argparse.ArgumentParser(description='Relabel dataset using YOLO classifier')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--dataset_path', type=str, help='Path to input YOLO dataset')
    parser.add_argument('--output_dir', type=str, help='Output directory for relabeled dataset')
    parser.add_argument('--classifier_model', type=str, help='Path to YOLO classifier model (.pt file)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5, help='Confidence threshold for predictions (default: 0.5)')
    parser.add_argument('--crop_padding', type=float, default=0.0, help='Padding around crops (default: 0.0)')
    parser.add_argument('--min_crop_size', type=int, default=0, help='Minimum crop size (default: 0)')
    parser.add_argument('--no_skip_low_confidence', action='store_false', dest='skip_low_confidence', help='Keep low confidence predictions')
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
    if args.classifier_model:
        config['classifier_model'] = args.classifier_model
    if args.confidence_threshold is not None:
        config['confidence_threshold'] = args.confidence_threshold
    if args.crop_padding is not None:
        config['crop_padding'] = args.crop_padding
    if args.min_crop_size is not None:
        config['min_crop_size'] = args.min_crop_size
    if 'skip_low_confidence' not in config:
        config['skip_low_confidence'] = args.skip_low_confidence
    
    # Validate required parameters
    required_params = ['dataset_path', 'output_dir', 'classifier_model']
    for param in required_params:
        if param not in config:
            logger.error(f"Missing required parameter: {param}")
            logger.info("Use --create_config to generate a configuration template")
            return
    
    # Check if classifier model exists
    if not Path(config['classifier_model']).exists():
        logger.error(f"Classifier model not found: {config['classifier_model']}")
        return
    
    # Create relabeler and process dataset
    relabeler = DatasetRelabeler(config)
    relabeler.relabel_dataset()

if __name__ == "__main__":
    main()
