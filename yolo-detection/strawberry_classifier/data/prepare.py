import os
import shutil
import random
from pathlib import Path
from typing import Union, Tuple
from .utils import get_image_paths, save_yaml, load_json


def prepare_dataset(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
    seed: int = 42
) -> None:
    """Prepare dataset from raw format with train/val/test splits"""
    
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    train_ratio, val_ratio, test_ratio = split_ratios
    
    if abs(sum(split_ratios) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    random.seed(seed)
    
    images_dir = source_dir / 'images'
    labels_dir = source_dir / 'labels'
    
    if not images_dir.exists():
        raise ValueError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise ValueError(f"Labels directory not found: {labels_dir}")
    
    image_files = get_image_paths(images_dir)
    
    valid_image_files = []
    for image_path in image_files:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            valid_image_files.append(image_path)
    
    if not valid_image_files:
        raise ValueError("No valid image-label pairs found")
    
    random.shuffle(valid_image_files)
    
    n_total = len(valid_image_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = [
        ('train', valid_image_files[:n_train]),
        ('val', valid_image_files[n_train:n_train + n_val]),
        ('test', valid_image_files[n_train + n_val:])
    ]
    
    for split_name, split_files in splits:
        split_dir = output_dir / split_name
        (split_dir / 'images').mkdir(parents=True, exist_ok=True)
        (split_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
        for image_path in split_files:
            label_path = labels_dir / f"{image_path.stem}.txt"
            
            shutil.copy2(image_path, split_dir / 'images' / image_path.name)
            shutil.copy2(label_path, split_dir / 'labels' / label_path.name)
    
    _copy_metadata(source_dir, output_dir)
    _create_data_yaml(output_dir, source_dir)


def _copy_metadata(source_dir: Path, output_dir: Path) -> None:
    """Copy metadata files from source to output"""
    for file_name in ['classes.txt', 'notes.json']:
        source_file = source_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, output_dir / file_name)


def _create_data_yaml(output_dir: Path, source_dir: Path) -> None:
    """Create YOLO data configuration file"""
    
    class_names = []
    
    classes_file = output_dir / 'classes.txt'
    if classes_file.exists():
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f if line.strip()]
    else:
        notes_file = output_dir / 'notes.json'
        if notes_file.exists():
            notes = load_json(notes_file)
            if 'categories' in notes:
                class_names = [cat['name'] for cat in notes['categories']]
    
    if not class_names:
        max_class = 0
        for split in ['train', 'val', 'test']:
            labels_dir = output_dir / split / 'labels'
            if labels_dir.exists():
                for label_file in labels_dir.glob('*.txt'):
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                max_class = max(max_class, class_id)
        class_names = [f"class_{i}" for i in range(max_class + 1)]
    
    data_config = {
        'train': str(output_dir / 'train' / 'images'),
        'val': str(output_dir / 'val' / 'images'),
        'test': str(output_dir / 'test' / 'images'),
        'nc': len(class_names),
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    save_yaml(data_config, output_dir / 'data.yaml')
