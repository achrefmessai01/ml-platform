"""
Utility functions for the strawberry classifier project
"""

import os
from pathlib import Path
import yaml
import json
from typing import List, Tuple, Dict, Optional, Any

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

def load_yaml(file_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict, file_path: str):
    """Save data to YAML file"""
    with open(file_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

def load_json(file_path: str) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict, file_path: str):
    """Save data to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_image_paths(directory: str, extensions: List[str] = None) -> List[str]:
    """Get all image file paths from a directory"""
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(Path(directory).glob(f'*{ext}'))
        image_paths.extend(Path(directory).glob(f'*{ext.upper()}'))
    
    return [str(p) for p in sorted(image_paths)]

def convert_bbox_format(bbox: List[float], img_width: int, img_height: int, 
                       from_format: str = 'yolo', to_format: str = 'xyxy') -> List[float]:
    """
    Convert bounding box format
    
    Args:
        bbox: Bounding box coordinates
        img_width: Image width
        img_height: Image height
        from_format: Source format ('yolo', 'xyxy', 'xywh')
        to_format: Target format ('yolo', 'xyxy', 'xywh')
    
    Returns:
        Converted bounding box coordinates
    """
    if from_format == 'yolo' and to_format == 'xyxy':
        # YOLO format: [x_center, y_center, width, height] (normalized)
        # XYXY format: [x1, y1, x2, y2] (absolute)
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)
        return [x1, y1, x2, y2]
    
    elif from_format == 'xyxy' and to_format == 'yolo':
        # XYXY format: [x1, y1, x2, y2] (absolute)
        # YOLO format: [x_center, y_center, width, height] (normalized)
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return [x_center, y_center, width, height]
    
    # Add more format conversions as needed
    return bbox

def visualize_annotations(image_path: str, label_path: str, class_names: List[str], 
                         output_path: Optional[str] = None) -> Any:
    """
    Visualize YOLO annotations on an image (supports both regular bbox and OBB)
    
    Args:
        image_path: Path to the image file
        label_path: Path to the label file
        class_names: List of class names
        output_path: Optional path to save the visualization
    
    Returns:
        Annotated image as numpy array
    """
    if not CV2_AVAILABLE:
        raise ImportError("OpenCV (cv2) is required for visualization. Install with: pip install opencv-python")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_height, img_width = image.shape[:2]
    
    # Load annotations
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            annotations = f.readlines()
        
        for annotation in annotations:
            parts = annotation.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                
                # Check if this is OBB format (9 values: class + 8 coordinates)
                if len(parts) == 9:
                    # YOLO OBB format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized)
                    coords = [float(x) for x in parts[1:9]]
                    
                    # Convert normalized coordinates to absolute pixels
                    points = []
                    for i in range(0, 8, 2):
                        x = int(coords[i] * img_width)
                        y = int(coords[i + 1] * img_height)
                        points.append((x, y))
                    
                    # Draw oriented bounding box
                    pts = np.array(points, np.int32)
                    cv2.polylines(image, [pts], True, (0, 255, 0), 2)
                    
                    # Draw class label at first point
                    if class_id < len(class_names):
                        label = class_names[class_id]
                        cv2.putText(image, label, points[0], 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                elif len(parts) >= 5:
                    # Regular YOLO bbox format: class_id x_center y_center width height (normalized)
                    bbox = [float(x) for x in parts[1:5]]
                    
                    # Convert YOLO format to XYXY
                    x1, y1, x2, y2 = convert_bbox_format(bbox, img_width, img_height, 'yolo', 'xyxy')
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Draw bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw class label
                    if class_id < len(class_names):
                        label = class_names[class_id]
                        cv2.putText(image, label, (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save visualization if output path is provided
    if output_path:
        cv2.imwrite(output_path, image)
    
    return image

def validate_dataset_structure(data_dir: str) -> Dict[str, int]:
    """
    Validate dataset structure and return statistics
    
    Args:
        data_dir: Path to the dataset directory
    
    Returns:
        Dictionary with dataset statistics
    """
    stats = {}
    
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            images_dir = os.path.join(split_dir, 'images')
            labels_dir = os.path.join(split_dir, 'labels')
            
            if os.path.exists(images_dir):
                image_files = get_image_paths(images_dir)
                stats[f'{split}_images'] = len(image_files)
            else:
                stats[f'{split}_images'] = 0
            
            if os.path.exists(labels_dir):
                label_files = list(Path(labels_dir).glob('*.txt'))
                stats[f'{split}_labels'] = len(label_files)
            else:
                stats[f'{split}_labels'] = 0
        else:
            stats[f'{split}_images'] = 0
            stats[f'{split}_labels'] = 0
    
    return stats
