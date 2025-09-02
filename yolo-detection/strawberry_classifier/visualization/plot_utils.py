"""
Visualization utilities for YOLO oriented bounding boxes and detections.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict
from matplotlib.colors import ListedColormap
import random


def plot_yolo_obb(
    image_path: Union[str, Path], 
    label_path: Union[str, Path],
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    show_labels: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.7,
    alpha: float = 0.8,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot YOLO oriented bounding boxes on an image.
    
    Args:
        image_path: Path to the image file
        label_path: Path to the YOLO OBB label file
        class_names: List of class names. If None, uses class IDs
        colors: List of RGB colors for each class. If None, generates random colors
        show_labels: Whether to show class labels on bounding boxes
        line_thickness: Thickness of bounding box lines
        font_scale: Scale factor for text
        alpha: Transparency of bounding boxes (0-1)
        save_path: Path to save the plotted image. If None, displays the image
        figsize: Figure size (width, height)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Read labels
    if not Path(label_path).exists():
        print(f"Warning: Label file {label_path} not found. Showing image without annotations.")
        labels = []
    else:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        labels = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 9:  # class_id + 8 coordinates
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:9]]
                labels.append((class_id, coords))
    
    # Generate colors if not provided
    if colors is None:
        num_classes = max([label[0] for label in labels]) + 1 if labels else 1
        colors = []
        for i in range(num_classes):
            # Generate bright, distinguishable colors
            random.seed(i * 42)  # Consistent colors
            colors.append((
                random.randint(100, 255),
                random.randint(100, 255), 
                random.randint(100, 255)
            ))
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip y-axis to match image coordinates
    ax.set_aspect('equal')
    
    # Plot oriented bounding boxes
    for class_id, coords in labels:
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = []
        for i in range(0, 8, 2):
            x = coords[i] * width
            y = coords[i + 1] * height
            pixel_coords.extend([x, y])
        
        # Create polygon from 4 corner points
        points = np.array(pixel_coords).reshape(4, 2)
        
        # Get color for this class
        color = np.array(colors[class_id % len(colors)]) / 255.0
        
        # Create polygon patch
        polygon = patches.Polygon(
            points, 
            linewidth=line_thickness, 
            edgecolor=color, 
            facecolor=(*color, alpha * 0.3),
            alpha=alpha
        )
        ax.add_patch(polygon)
        
        # Add class label if requested
        if show_labels:
            # Calculate center of bounding box for label placement
            center_x = np.mean(points[:, 0])
            center_y = np.mean(points[:, 1])
            
            # Determine label text
            if class_names and class_id < len(class_names):
                label_text = class_names[class_id]
            else:
                label_text = f"Class {class_id}"
            
            # Add text with background
            ax.text(
                center_x, center_y, label_text,
                fontsize=int(12 * font_scale),
                color='white',
                weight='bold',
                ha='center', va='center',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor=color,
                    alpha=0.8,
                    edgecolor='none'
                )
            )
    
    # Set title and remove axes
    ax.set_title(f"Image: {Path(image_path).name}", fontsize=14, weight='bold')
    ax.axis('off')
    
    # Save or display
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def plot_detections_with_obb(
    image_path: Union[str, Path],
    detections: List[Dict],
    class_names: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    confidence_threshold: float = 0.5,
    show_confidence: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.7,
    alpha: float = 0.8,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Plot detection results with oriented bounding boxes.
    
    Args:
        image_path: Path to the image file
        detections: List of detection dictionaries with keys:
                   - 'class_id': int
                   - 'confidence': float
                   - 'obb_coords': list of 8 normalized coordinates [x1,y1,x2,y2,x3,y3,x4,y4]
        class_names: List of class names. If None, uses class IDs
        colors: List of RGB colors for each class. If None, generates random colors
        confidence_threshold: Minimum confidence to display detection
        show_confidence: Whether to show confidence scores in labels
        line_thickness: Thickness of bounding box lines
        font_scale: Scale factor for text
        alpha: Transparency of bounding boxes (0-1)
        save_path: Path to save the plotted image. If None, displays the image
        figsize: Figure size (width, height)
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Filter detections by confidence
    filtered_detections = [
        det for det in detections 
        if det.get('confidence', 1.0) >= confidence_threshold
    ]
    
    # Generate colors if not provided
    if colors is None:
        max_class = max([det['class_id'] for det in filtered_detections]) if filtered_detections else 0
        num_classes = max_class + 1
        colors = []
        for i in range(num_classes):
            random.seed(i * 42)  # Consistent colors
            colors.append((
                random.randint(100, 255),
                random.randint(100, 255), 
                random.randint(100, 255)
            ))
    
    # Create figure and axis
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image)
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # Flip y-axis to match image coordinates
    ax.set_aspect('equal')
    
    # Plot detections
    for detection in filtered_detections:
        class_id = detection['class_id']
        confidence = detection.get('confidence', 1.0)
        coords = detection['obb_coords']
        
        # Convert normalized coordinates to pixel coordinates
        pixel_coords = []
        for i in range(0, 8, 2):
            x = coords[i] * width
            y = coords[i + 1] * height
            pixel_coords.extend([x, y])
        
        # Create polygon from 4 corner points
        points = np.array(pixel_coords).reshape(4, 2)
        
        # Get color for this class
        color = np.array(colors[class_id % len(colors)]) / 255.0
        
        # Create polygon patch
        polygon = patches.Polygon(
            points, 
            linewidth=line_thickness, 
            edgecolor=color, 
            facecolor=(*color, alpha * 0.3),
            alpha=alpha
        )
        ax.add_patch(polygon)
        
        # Add label
        # Calculate center of bounding box for label placement
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        
        # Determine label text
        if class_names and class_id < len(class_names):
            class_name = class_names[class_id]
        else:
            class_name = f"Class {class_id}"
        
        if show_confidence:
            label_text = f"{class_name}: {confidence:.2f}"
        else:
            label_text = class_name
        
        # Add text with background
        ax.text(
            center_x, center_y, label_text,
            fontsize=int(12 * font_scale),
            color='white',
            weight='bold',
            ha='center', va='center',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor=color,
                alpha=0.8,
                edgecolor='none'
            )
        )
    
    # Set title and remove axes
    ax.set_title(f"Detections: {Path(image_path).name} ({len(filtered_detections)} objects)", 
                fontsize=14, weight='bold')
    ax.axis('off')
    
    # Save or display
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)


def create_obb_from_corners(corners: np.ndarray, normalized: bool = True) -> List[float]:
    """
    Create YOLO OBB format from 4 corner points.
    
    Args:
        corners: Array of shape (4, 2) containing [x, y] coordinates of 4 corners
        normalized: Whether coordinates are normalized (0-1) or pixel coordinates
        
    Returns:
        List of 8 coordinates in YOLO OBB format [x1,y1,x2,y2,x3,y3,x4,y4]
    """
    return corners.flatten().tolist()


def convert_bbox_to_obb(bbox: List[float], angle: float = 0.0) -> List[float]:
    """
    Convert axis-aligned bounding box to oriented bounding box.
    
    Args:
        bbox: [x_center, y_center, width, height] in normalized coordinates
        angle: Rotation angle in degrees
        
    Returns:
        List of 8 coordinates in YOLO OBB format
    """
    x_center, y_center, width, height = bbox
    
    # Calculate half dimensions
    w_half = width / 2
    h_half = height / 2
    
    # Define corners relative to center
    corners = np.array([
        [-w_half, -h_half],  # top-left
        [w_half, -h_half],   # top-right
        [w_half, h_half],    # bottom-right
        [-w_half, h_half]    # bottom-left
    ])
    
    # Apply rotation if needed
    if angle != 0:
        angle_rad = np.radians(angle)
        rotation_matrix = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        corners = corners @ rotation_matrix.T
    
    # Translate to center position
    corners[:, 0] += x_center
    corners[:, 1] += y_center
    
    return corners.flatten().tolist()


def visualize_dataset_obb_samples(
    dataset_path: Union[str, Path],
    num_samples: int = 6,
    class_names: Optional[List[str]] = None,
    split: str = 'train',
    random_seed: int = 42,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> None:
    """
    Visualize random samples from an OBB dataset.
    
    Args:
        dataset_path: Path to dataset root directory
        num_samples: Number of samples to visualize
        class_names: List of class names
        split: Dataset split to sample from ('train', 'val', 'test')
        random_seed: Random seed for reproducible sampling
        save_path: Path to save the visualization
        figsize: Figure size
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / split / 'images'
    labels_dir = dataset_path / split / 'labels'
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(images_dir.glob(f"*{ext}")))
        image_files.extend(list(images_dir.glob(f"*{ext.upper()}")))
    
    if len(image_files) == 0:
        raise ValueError(f"No image files found in {images_dir}")
    
    # Sample random images
    random.seed(random_seed)
    sampled_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Calculate grid dimensions
    cols = min(3, num_samples)
    rows = (num_samples + cols - 1) // cols
    
    # Create subplot grid
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    for idx, image_path in enumerate(sampled_images):
        # Find corresponding label file
        label_path = labels_dir / f"{image_path.stem}.txt"
        
        # Load and display image
        image = cv2.imread(str(image_path))
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            
            ax = axes[idx]
            ax.imshow(image)
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)
            ax.set_aspect('equal')
            
            # Plot OBB annotations if label file exists
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        class_id = int(parts[0])
                        coords = [float(x) for x in parts[1:9]]
                        
                        # Convert to pixel coordinates
                        pixel_coords = []
                        for i in range(0, 8, 2):
                            x = coords[i] * width
                            y = coords[i + 1] * height
                            pixel_coords.extend([x, y])
                        
                        points = np.array(pixel_coords).reshape(4, 2)
                        
                        # Generate consistent color
                        random.seed(class_id * 42)
                        color = np.array([
                            random.randint(100, 255),
                            random.randint(100, 255), 
                            random.randint(100, 255)
                        ]) / 255.0
                        
                        # Create polygon
                        polygon = patches.Polygon(
                            points, 
                            linewidth=2, 
                            edgecolor=color, 
                            facecolor=(*color, 0.2),
                            alpha=0.8
                        )
                        ax.add_patch(polygon)
                        
                        # Add label
                        center_x = np.mean(points[:, 0])
                        center_y = np.mean(points[:, 1])
                        
                        if class_names and class_id < len(class_names):
                            label_text = class_names[class_id]
                        else:
                            label_text = f"C{class_id}"
                        
                        ax.text(
                            center_x, center_y, label_text,
                            fontsize=8, color='white', weight='bold',
                            ha='center', va='center',
                            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.8)
                        )
            
            ax.set_title(f"{image_path.name}", fontsize=10)
            ax.axis('off')
    
    # Hide unused subplots
    for idx in range(num_samples, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved dataset visualization to {save_path}")
    else:
        plt.show()
    
    plt.close(fig)
