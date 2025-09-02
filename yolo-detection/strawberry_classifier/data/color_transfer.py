import cv2
import numpy as np
import random
from pathlib import Path
import shutil
from tqdm import tqdm
from typing import Union, Optional


def extract_rgb_stats(image_path: Union[str, Path]) -> tuple:
    """Extract RGB statistics from an image"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    img_float = img.astype(np.float32) / 255.0
    mean_b = np.mean(img_float[:,:,0])
    mean_g = np.mean(img_float[:,:,1])
    mean_r = np.mean(img_float[:,:,2])
    
    if mean_g > 0:
        return (mean_b / mean_g, 1.0, mean_r / mean_g)
    return (1.0, 1.0, 1.0)


def get_random_reference_image(reference_path: Union[str, Path]) -> str:
    """Get a random reference image from a file or directory"""
    reference_path = Path(reference_path)
    
    if reference_path.is_file():
        return str(reference_path)
    elif reference_path.is_dir():
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(reference_path.glob(f'*{ext}'))
            image_files.extend(reference_path.glob(f'*{ext.upper()}'))
        
        if image_files:
            return str(random.choice(image_files))
    
    raise ValueError(f"No valid images found in: {reference_path}")


def transfer_rgb_color(source_path: Union[str, Path], target_path: Union[str, Path], 
                      output_path: Optional[Union[str, Path]] = None, intensity: float = 1.0) -> np.ndarray:
    """Apply RGB color transfer from source to target image"""
    actual_source = get_random_reference_image(source_path)
    
    source_img = cv2.imread(actual_source)
    target_img = cv2.imread(str(target_path))
    
    if source_img is None or target_img is None:
        raise ValueError("Could not load images")
    
    source_cast = extract_rgb_stats(actual_source)
    target_cast = extract_rgb_stats(target_path)
    
    img_float = target_img.astype(np.float32)
    
    b_ratio = source_cast[0] / target_cast[0] if target_cast[0] > 0 else 1.0
    g_ratio = source_cast[1] / target_cast[1] if target_cast[1] > 0 else 1.0
    r_ratio = source_cast[2] / target_cast[2] if target_cast[2] > 0 else 1.0
    
    b_ratio = 1.0 + (b_ratio - 1.0) * intensity
    g_ratio = 1.0 + (g_ratio - 1.0) * intensity
    r_ratio = 1.0 + (r_ratio - 1.0) * intensity
    
    b_ratio = np.clip(b_ratio, 0.3, 3.0)
    g_ratio = np.clip(g_ratio, 0.3, 3.0)
    r_ratio = np.clip(r_ratio, 0.3, 3.0)
    
    img_float[:,:,0] *= b_ratio
    img_float[:,:,1] *= g_ratio
    img_float[:,:,2] *= r_ratio
    
    img_float = np.clip(img_float, 0, 255)
    result_img = img_float.astype(np.uint8)
    
    if output_path:
        cv2.imwrite(str(output_path), result_img)
    
    return result_img


def apply_color_transfer_to_dataset(dataset_path: Union[str, Path], source_reference: Union[str, Path], 
                                   output_path: Union[str, Path], probability: float = 0.5, 
                                   intensity: float = 1.0) -> dict:
    """Apply color transfer to an entire dataset"""
    from .utils import load_yaml, save_yaml
    
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    
    source_path = Path(source_reference)
    if not source_path.exists():
        raise ValueError(f"Reference path does not exist: {source_reference}")
    
    splits = ['train', 'val', 'test']
    processed_count = 0
    skipped_count = 0
    
    for split in splits:
        images_dir = dataset_path / split / 'images'
        labels_dir = dataset_path / split / 'labels'
        
        if not images_dir.exists():
            continue
        
        output_images_dir = output_path / split / 'images'
        output_labels_dir = output_path / split / 'labels'
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        
        for image_file in tqdm(image_files, desc=f"Processing {split}"):
            output_image_path = output_images_dir / image_file.name
            
            if random.random() < probability:
                transfer_rgb_color(source_reference, str(image_file), str(output_image_path), intensity)
                processed_count += 1
            else:
                shutil.copy2(image_file, output_image_path)
                skipped_count += 1
            
            label_file = labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                output_label_path = output_labels_dir / label_file.name
                shutil.copy2(label_file, output_label_path)
    
    # Copy dataset configuration files
    data_yaml = dataset_path / 'data.yaml'
    if data_yaml.exists():
        data = load_yaml(str(data_yaml))
        data['path'] = str(output_path)
        save_yaml(data, str(output_path / 'data.yaml'))
    
    classes_txt = dataset_path / 'classes.txt'
    if classes_txt.exists():
        shutil.copy2(classes_txt, output_path / 'classes.txt')
    
    return {'processed': processed_count, 'skipped': skipped_count}
