from pathlib import Path
from collections import defaultdict, Counter
from typing import Union, Dict, Set, List
from .utils import load_yaml


def get_image_files(dataset_path: Union[str, Path], split_name: str) -> Set[str]:
    """Get all image files from a dataset split"""
    images_dir = Path(dataset_path) / split_name / "images"
    if not images_dir.exists():
        return set()
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = set()
    
    for file_path in images_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            image_files.add(file_path.stem)
    
    return image_files


def get_dataset_splits(dataset_path: Union[str, Path]) -> List[str]:
    """Detect available splits in a dataset"""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        return []
    
    possible_splits = ['train', 'val', 'valid', 'test']
    available_splits = []
    
    for split in possible_splits:
        split_dir = dataset_path / split
        if split_dir.exists() and split_dir.is_dir():
            images_dir = split_dir / "images"
            if images_dir.exists():
                available_splits.append(split)
    
    return available_splits


def compare_datasets(dataset1_path: Union[str, Path], dataset2_path: Union[str, Path]) -> Dict:
    """Compare two datasets for image overlap"""
    splits1 = get_dataset_splits(dataset1_path)
    splits2 = get_dataset_splits(dataset2_path)
    
    if not splits1:
        raise ValueError(f"No valid splits found in dataset1: {dataset1_path}")
    if not splits2:
        raise ValueError(f"No valid splits found in dataset2: {dataset2_path}")
    
    results = defaultdict(dict)
    
    for split1 in splits1:
        images1 = get_image_files(dataset1_path, split1)
        
        for split2 in splits2:
            images2 = get_image_files(dataset2_path, split2)
            
            overlap = images1.intersection(images2)
            overlap_percentage = (len(overlap) / len(images1) * 100) if images1 else 0
            
            results[split1][split2] = {
                'dataset1_count': len(images1),
                'dataset2_count': len(images2),
                'overlap_count': len(overlap),
                'overlap_percentage': overlap_percentage
            }
    
    return results


def load_class_names(dataset_path: Union[str, Path]) -> List[str]:
    """Load class names from dataset"""
    dataset_path = Path(dataset_path)
    
    data_yaml_path = dataset_path / 'data.yaml'
    if data_yaml_path.exists():
        data = load_yaml(str(data_yaml_path))
        if 'names' in data:
            return data['names']
    
    classes_txt_path = dataset_path / 'classes.txt'
    if classes_txt_path.exists():
        with open(classes_txt_path, 'r') as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    
    return []


def count_labels_in_split(labels_dir: Union[str, Path]) -> Counter:
    """Count labels in a split directory"""
    labels_dir = Path(labels_dir)
    if not labels_dir.exists():
        return Counter()
    
    label_counts = Counter()
    
    for label_file in labels_dir.glob('*.txt'):
        with open(label_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    class_id = int(line.split()[0])
                    label_counts[class_id] += 1
    
    return label_counts


def analyze_dataset(dataset_path: Union[str, Path]) -> Dict:
    """Analyze dataset label distribution"""
    dataset_path = Path(dataset_path)
    
    if (dataset_path / 'train').exists():
        splits = ['train', 'val', 'test']
        is_processed = True
    else:
        splits = ['']
        is_processed = False
    
    class_names = load_class_names(dataset_path)
    
    split_counts = {}
    total_counts = Counter()
    
    for split in splits:
        if split:
            labels_dir = dataset_path / split / 'labels'
            split_name = split
        else:
            labels_dir = dataset_path / 'labels'
            split_name = 'all'
        
        counts = count_labels_in_split(labels_dir)
        split_counts[split_name] = counts
        total_counts.update(counts)
    
    return {
        'class_names': class_names,
        'split_counts': split_counts,
        'total_counts': total_counts,
        'is_processed': is_processed
    }
