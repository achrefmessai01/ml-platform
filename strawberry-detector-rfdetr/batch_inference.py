#!/usr/bin/env python3
"""
Batch inference script for RF-DETR detection + classification pipeline.

This script:
1. Runs RF-DETR detection on a dataset of images
2. Extracts bounding boxes from detections
3. Classifies extracted crops using a classification model
4. Saves results in COCO format
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from rfdetr import RFDETRLarge, RFDETRMedium


class ImageDataset(Dataset):
    """Dataset for loading images from a directory."""
    
    def __init__(self, image_paths: List[Path]):
        self.image_paths = image_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        return image, str(image_path)


class CropDataset(Dataset):
    """Dataset for loading cropped regions from images."""
    
    def __init__(self, crops: List[Tuple[Image.Image, Dict]], transform=None):
        self.crops = crops
        self.transform = transform
    
    def __len__(self):
        return len(self.crops)
    
    def __getitem__(self, idx):
        crop_image, metadata = self.crops[idx]
        if self.transform:
            crop_image = self.transform(crop_image)
        return crop_image, metadata


def collate_images(batch):
    """Custom collate function for image batches."""
    images, paths = zip(*batch)
    return list(images), list(paths)


def collate_crops(batch):
    """Custom collate function for crop batches."""
    crops, metadata = zip(*batch)
    return torch.stack(crops), list(metadata)


def load_detection_model(weights_path: str, model_size: str = "large"):
    """Load RF-DETR detection model."""
    print(f"Loading RF-DETR {model_size} model from {weights_path}...")
    if model_size.lower() == "large":
        model = RFDETRLarge(pretrain_weights=weights_path)
    elif model_size.lower() == "medium":
        model = RFDETRMedium(pretrain_weights=weights_path)
    else:
        raise ValueError(f"Unsupported model size: {model_size}")
    
    model.optimize_for_inference(compile=False)
    print("Detection model loaded successfully!")
    return model


def load_classification_model(weights_path: str, num_classes: int):
    """Load classification model using PyTorch Lightning."""
    print(f"Loading classification model from {weights_path}...")
    
    # Load EfficientNet-V2-M model (matching the classifier architecture)
    from torchvision import models
    import pytorch_lightning as pl
    
    # Define the Lightning module matching the classifier architecture
    class StrawberryClassifier(pl.LightningModule):
        def __init__(self, config: dict = None, num_classes: int = 6):
            super().__init__()
            if config is None:
                config = {"num_classes": num_classes, "dropout_rate": 0.3}
            self.save_hyperparameters()
            self.config = config
            
            # EfficientNet backbone (matching the exact architecture)
            self.model = models.efficientnet_v2_m(weights=None)
            # EfficientNet-V2-M has 1280 features before the classifier
            self.model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(p=config.get("dropout_rate", 0.3), inplace=True),
                torch.nn.Linear(1280, config.get("num_classes", num_classes)),
            )
            
            self.criterion = torch.nn.CrossEntropyLoss()
            self.validation_step_outputs = []
        
        def forward(self, x):
            return self.model(x)
    
    try:
        # Load the checkpoint
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
        
        # Extract config from hyperparameters if available
        if 'hyper_parameters' in checkpoint:
            config = checkpoint['hyper_parameters'].get('config', {})
            if 'num_classes' not in config:
                config['num_classes'] = num_classes
            if 'dropout_rate' not in config:
                config['dropout_rate'] = 0.3
        else:
            config = {"num_classes": num_classes, "dropout_rate": 0.3}
            
        # Create model instance
        model = StrawberryClassifier(config=config)
        
        # Load the state dict
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
                
    except Exception as e:
        print(f"Checkpoint loading failed: {e}")
        raise RuntimeError(
            f"Model loading failed: {e}\n"
            "If you're using a checkpoint from Python 3.10/3.11, please convert it first using Docker:\n"
            "  docker run --rm -v $(pwd):/workspace -w /workspace python:3.10-slim bash -c \\\n"
            "    'pip install -q torch torchvision pytorch-lightning cloudpickle && \\\n"
            "     python -c \"import torch; c=torch.load(\\\"weights/model.pth\\\", map_location=\\\"cpu\\\"); \\\n"
            "     torch.save({\\\"state_dict\\\": c.state_dict(), \\\"hyper_parameters\\\": dict(c.hparams)}, \\\n"
            "     \\\"weights/model_converted.pth\\\")\"'\n"
            "Then use: --classifier-weights weights/model_converted.pth"
        )
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Classification model loaded successfully on {device}!")
    return model, device


def get_image_paths(dataset_path: Path, split: str = "train") -> List[Path]:
    """Get all image paths from a dataset directory."""
    # Try common dataset structures
    possible_paths = [
        dataset_path / split / "images",
        dataset_path / split,
        dataset_path / "images",
        dataset_path,
    ]
    
    for path in possible_paths:
        if path.exists() and path.is_dir():
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            image_paths = [
                p for p in path.iterdir()
                if p.is_file() and p.suffix.lower() in image_extensions
            ]
            if image_paths:
                print(f"Found {len(image_paths)} images in {path}")
                return sorted(image_paths)
    
    raise ValueError(f"Could not find images in {dataset_path}")


def run_detection_batch(
    model,
    images: List[Image.Image],
    image_paths: List[str],
    confidence_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Run detection on a batch of images."""
    results = []
    
    for image, image_path in zip(images, image_paths):
        detections = model.predict(image, threshold=confidence_threshold)
        width, height = image.size
        
        result = {
            'image_path': image_path,
            'image_width': width,
            'image_height': height,
            'boxes': detections.xyxy.tolist() if len(detections) > 0 else [],
            'scores': detections.confidence.tolist() if len(detections) > 0 else [],
            'class_ids': detections.class_id.tolist() if len(detections) > 0 else [],
        }
        results.append(result)
    
    return results


def extract_crops(
    image: Image.Image,
    boxes: List[List[float]],
    min_size: int = 32
) -> List[Image.Image]:
    """Extract cropped regions from an image based on bounding boxes."""
    crops = []
    
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        
        # Ensure valid crop dimensions
        width = x2 - x1
        height = y2 - y1
        
        if width >= min_size and height >= min_size:
            crop = image.crop((x1, y1, x2, y2))
            crops.append(crop)
        else:
            # Skip crops that are too small
            crops.append(None)
    
    return crops


def run_classification_batch(
    model,
    crops_tensor: torch.Tensor,
    device: torch.device,
    class_names: List[str] = None
) -> List[Dict[str, Any]]:
    """Run classification on a batch of crops."""
    crops_tensor = crops_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(crops_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        predicted_classes = torch.argmax(probabilities, dim=1)
    
    results = []
    for i in range(len(crops_tensor)):
        class_id = predicted_classes[i].item()
        confidence = probabilities[i][class_id].item()
        
        result = {
            'class_id': class_id,
            'class_name': class_names[class_id] if class_names else str(class_id),
            'confidence': confidence,
            'probabilities': probabilities[i].cpu().numpy().tolist()
        }
        results.append(result)
    
    return results


def convert_to_coco_format(
    detection_results: List[Dict[str, Any]],
    detection_categories: List[str] = None,
    classification_categories: List[str] = None
) -> Dict[str, Any]:
    """Convert detection results to COCO format."""
    
    # Initialize COCO structure
    coco_output = {
        "info": {
            "description": "RF-DETR Detection with Classification Results",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat()
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Add detection categories
    if detection_categories:
        for cat_id, cat_name in enumerate(detection_categories):
            coco_output["categories"].append({
                "id": cat_id,
                "name": cat_name,
                "supercategory": "object"
            })
    
    # Process each image
    annotation_id = 1
    
    for image_id, result in enumerate(detection_results, start=1):
        image_path = Path(result['image_path'])
        
        # Add image info
        image_info = {
            "id": image_id,
            "file_name": image_path.name,
            "width": result['image_width'],
            "height": result['image_height'],
            "path": str(image_path)
        }
        coco_output["images"].append(image_info)
        
        # Add annotations for each detection
        for i, (box, score, class_id) in enumerate(zip(
            result['boxes'],
            result['scores'],
            result['class_ids']
        )):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "bbox": [x1, y1, width, height],  # COCO format: [x, y, width, height]
                "area": width * height,
                "segmentation": [],
                "iscrowd": 0,
                "score": float(score)
            }
            
            # Add classification results if available
            if 'classifications' in result and result['classifications']:
                for classification in result['classifications']:
                    if classification['crop_index'] == i:
                        annotation['classification'] = {
                            "category_id": classification['classification_class_id'],
                            "category_name": classification['classification_class_name'],
                            "confidence": classification['classification_confidence']
                        }
                        break
            
            coco_output["annotations"].append(annotation)
            annotation_id += 1
    
    return coco_output


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference with RF-DETR detection and classification"
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        required=True,
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--detection-weights',
        type=str,
        required=True,
        help='Path to RF-DETR detection model weights'
    )
    parser.add_argument(
        '--classifier-weights',
        type=str,
        default=None,
        help='Path to classification model weights (optional)'
    )
    parser.add_argument(
        '--model-size',
        type=str,
        default='large',
        choices=['large', 'medium'],
        help='RF-DETR model size'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Dataset split to use (train/val/test)'
    )
    parser.add_argument(
        '--detection-batch-size',
        type=int,
        default=4,
        help='Batch size for detection'
    )
    parser.add_argument(
        '--classification-batch-size',
        type=int,
        default=32,
        help='Batch size for classification'
    )
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for detections'
    )
    parser.add_argument(
        '--num-classes',
        type=int,
        default=6,
        help='Number of classification classes'
    )
    parser.add_argument(
        '--output-file',
        type=str,
        default='inference_results.json',
        help='Output JSON file for results in COCO format'
    )
    parser.add_argument(
        '--detection-class-names',
        type=str,
        nargs='+',
        default=None,
        help='List of class names for detection'
    )
    parser.add_argument(
        '--class-names',
        type=str,
        nargs='+',
        default=None,
        help='List of class names for classification'
    )
    
    args = parser.parse_args()
    
    # Setup
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Load detection model
    detection_model = load_detection_model(args.detection_weights, args.model_size)
    
    # Get image paths
    image_paths = get_image_paths(dataset_path, args.split)
    
    # Create dataset and dataloader for detection
    image_dataset = ImageDataset(image_paths)
    detection_loader = DataLoader(
        image_dataset,
        batch_size=args.detection_batch_size,
        shuffle=False,
        collate_fn=collate_images,
        num_workers=0  # Keep 0 for PIL image compatibility
    )
    
    # Run detection on all images
    print(f"\n{'='*60}")
    print(f"Running detection on {len(image_paths)} images...")
    print(f"{'='*60}\n")
    
    all_detection_results = []
    all_crops_data = []
    
    for images, paths in tqdm(detection_loader, desc="Detection", unit="batch"):
        batch_results = run_detection_batch(
            detection_model,
            images,
            paths,
            args.confidence_threshold
        )
        all_detection_results.extend(batch_results)
        
        # Extract crops for classification
        for image, result in zip(images, batch_results):
            if len(result['boxes']) > 0:
                crops = extract_crops(image, result['boxes'])
                for i, (crop, box, score) in enumerate(zip(
                    crops,
                    result['boxes'],
                    result['scores']
                )):
                    if crop is not None:
                        metadata = {
                            'image_path': result['image_path'],
                            'detection_box': box,
                            'detection_score': score,
                            'detection_class_id': result['class_ids'][i],
                            'crop_index': i
                        }
                        all_crops_data.append((crop, metadata))
    
    print(f"\nDetection complete! Found {len(all_crops_data)} objects across all images.")
    
    # Run classification if model is provided
    if args.classifier_weights and len(all_crops_data) > 0:
        print(f"\n{'='*60}")
        print(f"Running classification on {len(all_crops_data)} crops...")
        print(f"{'='*60}\n")
        
        # Load classification model
        classifier, device = load_classification_model(
            args.classifier_weights,
            args.num_classes
        )
        
        # Prepare transform for classification
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Create dataset and dataloader for classification
        crop_dataset = CropDataset(all_crops_data, transform=transform)
        classification_loader = DataLoader(
            crop_dataset,
            batch_size=args.classification_batch_size,
            shuffle=False,
            collate_fn=collate_crops,
            num_workers=0
        )
        
        # Run classification
        classification_results = []
        for crops_tensor, metadata_batch in tqdm(
            classification_loader,
            desc="Classification",
            unit="batch"
        ):
            batch_results = run_classification_batch(
                classifier,
                crops_tensor,
                device,
                args.class_names
            )
            
            # Combine with metadata
            for result, metadata in zip(batch_results, metadata_batch):
                combined = {**metadata, **result}
                classification_results.append(combined)
        
        # Merge classification results back into detection results
        for det_result in all_detection_results:
            det_result['classifications'] = []
            image_path = det_result['image_path']
            
            for class_result in classification_results:
                if class_result['image_path'] == image_path:
                    det_result['classifications'].append({
                        'crop_index': class_result['crop_index'],
                        'detection_box': class_result['detection_box'],
                        'detection_score': class_result['detection_score'],
                        'detection_class_id': class_result['detection_class_id'],
                        'classification_class_id': class_result['class_id'],
                        'classification_class_name': class_result['class_name'],
                        'classification_confidence': class_result['confidence']
                    })
        
        print(f"\nClassification complete!")
    
    # Convert to COCO format
    print(f"\nConverting results to COCO format...")
    coco_results = convert_to_coco_format(
        all_detection_results,
        detection_categories=args.detection_class_names,
        classification_categories=args.class_names
    )
    
    # Save results in COCO format
    output_path = Path(args.output_file)
    with open(output_path, 'w') as f:
        json.dump(coco_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}\n")
    
    # Print summary statistics
    total_detections = sum(len(r['boxes']) for r in all_detection_results)
    images_with_detections = sum(1 for r in all_detection_results if len(r['boxes']) > 0)
    
    print("Summary:")
    print(f"  Total images processed: {len(coco_results['images'])}")
    print(f"  Images with detections: {images_with_detections}")
    print(f"  Total annotations: {len(coco_results['annotations'])}")
    print(f"  Total categories: {len(coco_results['categories'])}")
    print(f"  Average detections per image: {total_detections / len(all_detection_results):.2f}")
    
    if args.classifier_weights:
        print(f"  Total crops classified: {len(classification_results)}")


if __name__ == "__main__":
    main()
