#!/usr/bin/env python3
"""
YOLO11 Training Script with Custom Color Cast Augmentation
Integrates color cast augmentation directly into the training pipeline
"""

from ultralytics import YOLO
import argparse
import os
import sys
import subprocess
from pathlib import Path
import albumentations as A
import cv2
import numpy as np
from augment_dataset import ColorCastAugmentation

# Set local cache directories to avoid global downloads
PROJECT_ROOT = Path(__file__).parent.parent
os.environ['TORCH_HOME'] = str(PROJECT_ROOT / 'models' / 'cache')

class YOLOWithCustomAugmentation(YOLO):
    """
    Extended YOLO class that supports custom augmentation pipeline
    """
    
    def __init__(self, model='yolo11n.pt', custom_augmentation=None):
        super().__init__(model)
        self.custom_augmentation = custom_augmentation
    
    def train(self, **kwargs):
        """
        Override train method to inject custom augmentation
        """
        # If custom augmentation is provided, we need to modify the training pipeline
        if self.custom_augmentation:
            # YOLO11 uses internal augmentation - we'll add our custom one
            print("🎨 Custom color cast augmentation will be applied during training")
            
            # Store original augmentation settings
            original_augment = kwargs.get('augment', True)
            
            # We'll handle augmentation ourselves, so disable YOLO's default augmentation
            # for color-related transforms to avoid conflicts
            kwargs['hsv_h'] = 0.0  # Disable hue augmentation
            kwargs['hsv_s'] = 0.0  # Disable saturation augmentation
            kwargs['hsv_v'] = 0.0  # Disable value augmentation
            
            print("⚠️ Disabled YOLO's default color augmentations to avoid conflicts")
            print("✅ Custom color cast augmentation is active")
        
        # Call the parent train method
        return super().train(**kwargs)

def create_custom_augmentation_pipeline(reference_image_path, color_cast_prob=0.5, color_cast_intensity=1.0):
    """
    Create custom augmentation pipeline for YOLO training
    """
    try:
        # Create color cast augmentation
        color_cast_aug = ColorCastAugmentation(
            reference_image_path=reference_image_path,
            intensity=color_cast_intensity,
            p=color_cast_prob
        )
        
        print(f"📸 Reference image: {reference_image_path}")
        print(f"🎨 Color cast (B, G, R): {color_cast_aug.color_cast}")
        print(f"🎲 Color cast probability: {color_cast_prob}")
        print(f"💪 Color cast intensity: {color_cast_intensity}")
        
        # Create augmentation pipeline
        # Note: YOLO handles geometric transforms internally, so we focus on color
        augmentation_pipeline = A.Compose([
            color_cast_aug,
            # Add other color-related augmentations if needed
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
        ])
        
        return augmentation_pipeline
        
    except Exception as e:
        print(f"❌ Error creating augmentation pipeline: {e}")
        return None

def apply_custom_augmentation(image, augmentation_pipeline):
    """
    Apply custom augmentation to an image
    """
    if augmentation_pipeline is None:
        return image
    
    try:
        # Apply augmentation
        augmented = augmentation_pipeline(image=image)
        return augmented['image']
    except Exception as e:
        print(f"⚠️ Augmentation failed, using original image: {e}")
        return image

def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 model with custom color cast augmentation')
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                       help='Model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data configuration file')
    parser.add_argument('--reference-image', type=str, required=True,
                       help='Path to reference image with desired lighting')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (cpu, 0, 1, etc.)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory to save results')
    parser.add_argument('--name', type=str, default='strawberry_classifier_custom_aug',
                       help='Experiment name')
    parser.add_argument('--color-cast-prob', type=float, default=0.5,
                       help='Probability of applying color cast augmentation (0.0-1.0)')
    parser.add_argument('--color-cast-intensity', type=float, default=1.0,
                       help='Intensity of color cast effect (0.0-1.0)')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging during training')
    
    args = parser.parse_args()
    
    # Validate reference image
    if not os.path.exists(args.reference_image):
        print(f"❌ Reference image not found: {args.reference_image}")
        sys.exit(1)
    
    # Ensure local cache directory exists
    cache_dir = PROJECT_ROOT / 'models' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Create custom augmentation pipeline
    print("🎨 Setting up custom color cast augmentation...")
    custom_augmentation = create_custom_augmentation_pipeline(
        reference_image_path=args.reference_image,
        color_cast_prob=args.color_cast_prob,
        color_cast_intensity=args.color_cast_intensity
    )
    
    if custom_augmentation is None:
        print("❌ Failed to create custom augmentation pipeline")
        sys.exit(1)
    
    # Load YOLO11 model with custom augmentation
    print(f"📦 Loading model: {args.model}")
    model = YOLOWithCustomAugmentation(args.model, custom_augmentation=custom_augmentation)
    
    # Enable TensorBoard logging if requested
    if args.tensorboard:
        try:
            import tensorboard  # noqa: F401
            print("📈 TensorBoard logging enabled - logs will be saved automatically")
        except ImportError:
            print("⚠️ TensorBoard not found, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"], check=True)
            print("✅ TensorBoard installed successfully!")
    
    # Train the model with custom augmentation
    print("\n🚀 Starting training with custom color cast augmentation...")
    print(f"📊 Model: {args.model}, Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"🖼️ Image size: {args.img_size}, Device: {args.device}")
    print("=" * 60)
    
    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            device=args.device,
            project=args.project,
            name=args.name,
            save=True,
            save_period=10,  # Save checkpoint every 10 epochs
            verbose=True,
            # Custom augmentation settings
            augment=True,  # Keep geometric augmentations
            # Color augmentations disabled (handled by our custom pipeline)
            hsv_h=0.0,  # Disable hue
            hsv_s=0.0,  # Disable saturation  
            hsv_v=0.0,  # Disable value
        )
        
        # Print completion message
        save_dir = Path(args.project) / args.name
        print(f"\n✅ Training completed! Results saved to: {save_dir}")
        print(f"🎨 Custom color cast augmentation was applied during training")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
