#!/usr/bin/env python3

import cv2
import numpy as np
import os
import sys
import argparse
import random
from pathlib import Path
import shutil
from tqdm import tqdm

CONFIG = {
    'intensity': 1.0,
    'brightness_min': 0.3,
    'brightness_max': 3.0,
}

def extract_rgb_stats(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_float = img.astype(np.float32) / 255.0
    mean_b = np.mean(img_float[:,:,0])
    mean_g = np.mean(img_float[:,:,1])
    mean_r = np.mean(img_float[:,:,2])
    
    if mean_g > 0:
        r_multiplier = mean_r / mean_g
        b_multiplier = mean_b / mean_g
        g_multiplier = 1.0
    else:
        r_multiplier = b_multiplier = g_multiplier = 1.0
    
    return (b_multiplier, g_multiplier, r_multiplier)

def get_random_reference_image(reference_path):
    """Get a random reference image from a file or directory"""
    reference_path = Path(reference_path)
    
    if reference_path.is_file():
        return str(reference_path)
    elif reference_path.is_dir():
        # Find all image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(reference_path.glob(f'*{ext}'))
            image_files.extend(reference_path.glob(f'*{ext.upper()}'))
        
        if not image_files:
            raise ValueError(f"No image files found in reference directory: {reference_path}")
        
        return str(random.choice(image_files))
    else:
        raise ValueError(f"Reference path does not exist: {reference_path}")

def transfer_rgb_color(source_path, target_path, output_path=None, intensity=1.0):
    # Select random reference if source_path is a directory
    actual_source = get_random_reference_image(source_path)
    
    source_img = cv2.imread(actual_source)
    target_img = cv2.imread(target_path)
    
    if source_img is None:
        raise ValueError(f"Could not load source image: {actual_source}")
    if target_img is None:
        raise ValueError(f"Could not load target image: {target_path}")
    
    source_cast = extract_rgb_stats(actual_source)
    target_cast = extract_rgb_stats(target_path)
    
    img_float = target_img.astype(np.float32)
    
    b_ratio = source_cast[0] / target_cast[0] if target_cast[0] > 0 else 1.0
    g_ratio = source_cast[1] / target_cast[1] if target_cast[1] > 0 else 1.0
    r_ratio = source_cast[2] / target_cast[2] if target_cast[2] > 0 else 1.0
    
    b_ratio = 1.0 + (b_ratio - 1.0) * intensity
    g_ratio = 1.0 + (g_ratio - 1.0) * intensity
    r_ratio = 1.0 + (r_ratio - 1.0) * intensity
    
    b_ratio = np.clip(b_ratio, CONFIG['brightness_min'], CONFIG['brightness_max'])
    g_ratio = np.clip(g_ratio, CONFIG['brightness_min'], CONFIG['brightness_max'])
    r_ratio = np.clip(r_ratio, CONFIG['brightness_min'], CONFIG['brightness_max'])
    
    img_float[:,:,0] = img_float[:,:,0] * b_ratio
    img_float[:,:,1] = img_float[:,:,1] * g_ratio
    img_float[:,:,2] = img_float[:,:,2] * r_ratio
    
    img_float = np.clip(img_float, 0, 255)
    result_img = img_float.astype(np.uint8)
    
    if output_path:
        cv2.imwrite(output_path, result_img)
    
    return result_img, (b_ratio, g_ratio, r_ratio, actual_source)

def process_dataset(dataset_path, source_reference, output_path, probability=0.5, intensity=1.0):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    # Validate reference path
    source_path = Path(source_reference)
    if not source_path.exists():
        raise ValueError(f"Reference path does not exist: {source_reference}")
    
    # Count available reference images if it's a directory
    if source_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        ref_images = []
        for ext in image_extensions:
            ref_images.extend(source_path.glob(f'*{ext}'))
            ref_images.extend(source_path.glob(f'*{ext.upper()}'))
        print(f"Found {len(ref_images)} reference images in {source_reference}")
    else:
        print(f"Using single reference image: {source_reference}")
    
    # Create output directory structure
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Copy non-image files
    for file in ['classes.txt', 'data.yaml', 'notes.json']:
        src_file = dataset_path / file
        if src_file.exists():
            shutil.copy2(src_file, output_path / file)
    
    # Process each split (train, val, test)
    for split in ['train', 'val', 'test']:
        split_dir = dataset_path / split
        if not split_dir.exists():
            continue
            
        output_split_dir = output_path / split
        output_split_dir.mkdir(exist_ok=True)
        
        # Copy labels directory
        labels_dir = split_dir / 'labels'
        if labels_dir.exists():
            shutil.copytree(labels_dir, output_split_dir / 'labels', dirs_exist_ok=True)
        
        # Copy labels.cache if it exists
        cache_file = split_dir / 'labels.cache'
        if cache_file.exists():
            shutil.copy2(cache_file, output_split_dir / 'labels.cache')
        
        # Process images
        images_dir = split_dir / 'images'
        if not images_dir.exists():
            continue
            
        output_images_dir = output_split_dir / 'images'
        output_images_dir.mkdir(exist_ok=True)
        
        image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        processed_count = 0
        total_count = len(image_files)
        
        # Progress bar for current split
        with tqdm(image_files, desc=f"Processing {split}", unit="img") as pbar:
            for img_file in pbar:
                output_file = output_images_dir / img_file.name
                
                if random.random() < probability:
                    # Apply color transfer
                    try:
                        result_img, info = transfer_rgb_color(source_reference, str(img_file), str(output_file), intensity)
                        processed_count += 1
                        pbar.set_postfix({"Processed": f"{processed_count}/{total_count}"})
                    except Exception as e:
                        # If transfer fails, copy original
                        shutil.copy2(img_file, output_file)
                        pbar.set_postfix({"Failed": img_file.name})
                else:
                    # Copy original image
                    shutil.copy2(img_file, output_file)
        
        print(f"{split}: {processed_count}/{total_count} images processed ({processed_count/total_count*100:.1f}%)")
    
    print(f"Dataset processing complete. Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='RGB color transfer between images or datasets')
    parser.add_argument('--source', type=str, required=True, help='Source image or directory with reference images')
    
    # Single image mode
    parser.add_argument('--target', type=str, help='Target image to modify')
    parser.add_argument('--output', type=str, help='Output path')
    
    # Dataset mode
    parser.add_argument('--dataset', type=str, help='Path to dataset directory')
    parser.add_argument('--output-dataset', type=str, help='Output dataset directory')
    parser.add_argument('--probability', type=float, default=0.5, help='Probability of applying transformation (0.0-1.0)')
    
    parser.add_argument('--intensity', type=float, default=1.0, help='Intensity (0.0-1.0)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"Source not found: {args.source}")
        sys.exit(1)
    
    # Dataset mode
    if args.dataset:
        if not args.output_dataset:
            print("--output-dataset required when using --dataset")
            sys.exit(1)
        
        if not os.path.exists(args.dataset):
            print(f"Dataset not found: {args.dataset}")
            sys.exit(1)
        
        try:
            process_dataset(args.dataset, args.source, args.output_dataset, args.probability, args.intensity)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        return
    
    # Single image mode
    if not args.target:
        print("--target required for single image mode")
        sys.exit(1)
    
    if not os.path.exists(args.target):
        print(f"Target not found: {args.target}")
        sys.exit(1)
    
    if not args.output:
        base, ext = os.path.splitext(args.target)
        args.output = f"{base}_rgb_transferred{ext}"
    
    try:
        result_img, info = transfer_rgb_color(args.source, args.target, args.output, args.intensity)
        ratios = info[:3]  # First 3 elements are the ratios
        used_source = info[3]  # 4th element is the actual source image used
        print(f"RGB ratios (B, G, R): {ratios[0]:.3f}, {ratios[1]:.3f}, {ratios[2]:.3f}")
        print(f"Reference image used: {used_source}")
        print(f"Saved to: {args.output}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
