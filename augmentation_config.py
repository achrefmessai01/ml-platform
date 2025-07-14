
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
            reference_image_path="data/processed/farm_dataset/train/images/1b5bbf87-IMG_9602.JPEG",
            intensity=1.0,
            p=0.3
        ),
        
        # Keep values in proper range for YOLO
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0, p=1.0),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# For direct import
augmentation_pipeline = get_training_augmentations()
