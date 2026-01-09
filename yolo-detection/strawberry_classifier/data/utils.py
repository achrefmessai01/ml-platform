import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml


def load_yaml(file_path: Union[str, Path]) -> Dict:
    """Load YAML configuration file."""
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, file_path: Union[str, Path]) -> None:
    """Save data to YAML file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False)


def load_json(file_path: Union[str, Path]) -> Dict:
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data: Dict, file_path: Union[str, Path]) -> None:
    """Save data to JSON file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def get_image_paths(directory: Union[str, Path], extensions: Optional[List[str]] = None) -> List[Path]:
    """Get all image file paths from a directory."""
    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    directory = Path(directory)
    image_paths = []

    for ext in extensions:
        image_paths.extend(directory.glob(f"*{ext}"))
        image_paths.extend(directory.glob(f"*{ext.upper()}"))

    return sorted(image_paths)


def convert_bbox_format(
    bbox: List[float],
    img_width: int,
    img_height: int,
    from_format: str = "yolo",
    to_format: str = "xyxy",
) -> List[float]:
    """Convert bounding box format between YOLO and XYXY."""
    if from_format == "yolo" and to_format == "xyxy":
        x_center, y_center, width, height = bbox
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)
        return [x1, y1, x2, y2]

    elif from_format == "xyxy" and to_format == "yolo":
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        return [x_center, y_center, width, height]

    return bbox


def validate_dataset_structure(data_dir: Union[str, Path]) -> Dict[str, int]:
    """Validate dataset structure and return statistics."""
    data_dir = Path(data_dir)
    stats = {}

    for split in ["train", "val", "test"]:
        images_dir = data_dir / split / "images"
        labels_dir = data_dir / split / "labels"

        stats[f"{split}_images"] = len(get_image_paths(images_dir)) if images_dir.exists() else 0
        stats[f"{split}_labels"] = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

    return stats
