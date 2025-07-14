#!/usr/bin/env python3
"""
Custom Color Cast Augmentation for YOLO Training
Integrates with YOLO's training pipeline for on-the-fly color cast augmentation
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

# Add the parent directory to the path so we can import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class ColorCastAugmentation(ImageOnlyTransform):
    """
    Custom albumentations transform for color cast augmentation.
    Applies color cast from a reference image to training images.
    """
    
    def __init__(self, reference_image_path, intensity=1.0, always_apply=False, p=0.5):
        """
        Args:
            reference_image_path: Path to reference image with desired color cast
            intensity: Intensity of the color cast effect (0.0 to 1.0)
            always_apply: Whether to always apply the transform
            p: Probability of applying the transform
        """
        super().__init__(always_apply, p)
        self.reference_image_path = reference_image_path
        self.intensity = intensity
        self.color_cast = self._extract_color_cast()
        
    def _extract_color_cast(self):
        """Extract color cast from reference image"""
        reference_img = cv2.imread(str(self.reference_image_path))
        if reference_img is None:
            raise ValueError(f"Could not load reference image: {self.reference_image_path}")
        
        # Convert to float and normalize
        img_float = reference_img.astype(np.float32) / 255.0
        
        # Calculate mean values for each channel (BGR in OpenCV)
        mean_b = np.mean(img_float[:,:,0])
        mean_g = np.mean(img_float[:,:,1]) 
        mean_r = np.mean(img_float[:,:,2])
        
        # Normalize relative to green channel (as reference)
        if mean_g > 0:
            r_multiplier = mean_r / mean_g
            b_multiplier = mean_b / mean_g
            g_multiplier = 1.0
        else:
            r_multiplier = b_multiplier = g_multiplier = 1.0
        
        return (b_multiplier, g_multiplier, r_multiplier)
    
    def apply(self, img, **params):
        """Apply color cast to image"""
        img_float = img.astype(np.float32)
        
        # Apply the color cast with intensity control
        img_float[:,:,0] = img_float[:,:,0] * (1 + (self.color_cast[0] - 1) * self.intensity)  # Blue
        img_float[:,:,1] = img_float[:,:,1] * (1 + (self.color_cast[1] - 1) * self.intensity)  # Green
        img_float[:,:,2] = img_float[:,:,2] * (1 + (self.color_cast[2] - 1) * self.intensity)  # Red
        
        # Clip values to valid range
        img_float = np.clip(img_float, 0, 255)
        
        return img_float.astype(np.uint8)
    
    def get_transform_init_args_names(self):
        return ("reference_image_path", "intensity")

def create_yolo_augmentation_pipeline(reference_image_path, color_cast_prob=0.5, color_cast_intensity=1.0):
    """
    Create an augmentation pipeline that includes color cast augmentation
    along with other useful augmentations for YOLO training.
    """
    
    # Create the augmentation pipeline
    transform = A.Compose([
        # Existing useful augmentations for object detection
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        
        # Custom color cast augmentation
        ColorCastAugmentation(
            reference_image_path=reference_image_path,
            intensity=color_cast_intensity,
            p=color_cast_prob
        ),
        
        # Normalize (keep values in 0-255 range for YOLO)
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    return transform

def save_augmentation_config(reference_image_path, output_path, color_cast_prob=0.5, color_cast_intensity=1.0):
    """
    Save augmentation configuration that can be imported by YOLO training.
    """
    
    config_content = f'''
"""
Custom YOLO Augmentation Configuration
Auto-generated augmentation pipeline with color cast support
"""

import albumentations as A
from scripts.augment_dataset import ColorCastAugmentation

def get_training_augmentations():
    """Get augmentation pipeline for YOLO training"""
    
    return A.Compose([
        # Standard augmentations
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=20,
            p=0.5
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.Blur(blur_limit=3, p=0.3),
        
        # Custom color cast augmentation
        ColorCastAugmentation(
            reference_image_path="{reference_image_path}",
            intensity={color_cast_intensity},
            p={color_cast_prob}
        ),
        
        # Keep values in proper range for YOLO
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# For direct import
augmentation_pipeline = get_training_augmentations()
'''
    
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Augmentation configuration saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Setup color cast augmentation for YOLO training')
    parser.add_argument('--reference', type=str, required=True,
                       help='Path to reference image with desired lighting')
    parser.add_argument('--output', type=str, default='augmentation_config.py',
                       help='Output path for augmentation configuration')
    parser.add_argument('--color-cast-prob', type=float, default=0.5,
                       help='Probability of applying color cast augmentation (0.0-1.0)')
    parser.add_argument('--color-cast-intensity', type=float, default=1.0,
                       help='Intensity of color cast effect (0.0-1.0)')
    
    args = parser.parse_args()
    
    # Validate reference image
    if not os.path.exists(args.reference):
        print(f"❌ Reference image not found: {args.reference}")
        sys.exit(1)
    
    try:
        # Test the color cast extraction
        color_cast_aug = ColorCastAugmentation(args.reference, p=1.0)
        print(f"📸 Reference image: {args.reference}")
        print(f"🎨 Color cast (B, G, R): {color_cast_aug.color_cast}")
        print(f"🎲 Color cast probability: {args.color_cast_prob}")
        print(f"💪 Color cast intensity: {args.color_cast_intensity}")
        
        # Save configuration
        save_augmentation_config(
            reference_image_path=args.reference,
            output_path=args.output,
            color_cast_prob=args.color_cast_prob,
            color_cast_intensity=args.color_cast_intensity
        )
        
        print("\n💡 Next steps:")
        print("1. Import this configuration in your YOLO training script")
        print("2. Use the augmentation pipeline during training")
        print("3. The color cast will be applied on-the-fly during training")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
