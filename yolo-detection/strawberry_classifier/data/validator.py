from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

from .utils import load_yaml


def validate_dataset_structure(dataset_path: Union[str, Path]) -> List[str]:
    """Validate dataset directory structure."""
    dataset_path = Path(dataset_path)
    issues = []

    if not dataset_path.exists():
        issues.append(f"Dataset path does not exist: {dataset_path}")
        return issues

    required_splits = ["train", "val"]
    for split in required_splits:
        split_dir = dataset_path / split
        if not split_dir.exists():
            issues.append(f"Missing required split: {split}")
            continue

        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"

        if not images_dir.exists():
            issues.append(f"Missing images directory: {images_dir}")
        if not labels_dir.exists():
            issues.append(f"Missing labels directory: {labels_dir}")

    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        issues.append("Missing data.yaml file")

    return issues


def validate_images_and_labels(
    dataset_path: Union[str, Path],
) -> Tuple[List[str], Dict[str, int]]:
    """Validate images and labels correspondence."""
    dataset_path = Path(dataset_path)
    issues = []
    stats = {"images": 0, "labels": 0, "orphaned_images": 0, "orphaned_labels": 0}

    for split in ["train", "val", "test"]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        if not images_dir.exists():
            continue

        image_files = set()
        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            image_files.update(f.stem for f in images_dir.glob(ext))

        label_files = set()
        if labels_dir.exists():
            label_files.update(f.stem for f in labels_dir.glob("*.txt"))

        stats["images"] += len(image_files)
        stats["labels"] += len(label_files)

        orphaned_images = image_files - label_files
        orphaned_labels = label_files - image_files

        stats["orphaned_images"] += len(orphaned_images)
        stats["orphaned_labels"] += len(orphaned_labels)

        if orphaned_images:
            issues.append(f"{split}: {len(orphaned_images)} images without labels")
        if orphaned_labels:
            issues.append(f"{split}: {len(orphaned_labels)} labels without images")

    return issues, stats


def validate_label_format(dataset_path: Union[str, Path]) -> Tuple[List[str], Counter]:
    """Validate label file format."""
    dataset_path = Path(dataset_path)
    issues = []
    class_counts = Counter()

    data_yaml = dataset_path / "data.yaml"
    num_classes = 0

    if data_yaml.exists():
        try:
            data = load_yaml(str(data_yaml))
            num_classes = data.get("nc", 0)
        except Exception:
            issues.append("Could not parse data.yaml")

    for split in ["train", "val", "test"]:
        labels_dir = dataset_path / split / "labels"

        if not labels_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            line_num = 0
            try:
                with open(label_file, "r") as f:
                    for line in f:
                        line_num += 1
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 5:
                            issues.append(f"{label_file}:{line_num} - Invalid format (need at least 5 values)")
                            continue

                        try:
                            class_id = int(parts[0])
                            coords = [float(x) for x in parts[1:5]]

                            if num_classes > 0 and class_id >= num_classes:
                                issues.append(
                                    f"{label_file}:{line_num} - Class ID {class_id} >= num_classes {num_classes}"
                                )

                            if not all(0 <= coord <= 1 for coord in coords):
                                issues.append(f"{label_file}:{line_num} - Coordinates not normalized (0-1)")

                            class_counts[class_id] += 1

                        except ValueError:
                            issues.append(f"{label_file}:{line_num} - Invalid number format")

            except Exception as e:
                issues.append(f"Could not read {label_file}: {e}")

    return issues, class_counts


def validate_dataset(dataset_path: Union[str, Path]) -> Dict:
    """Comprehensive dataset validation."""
    structure_issues = validate_dataset_structure(dataset_path)

    if structure_issues:
        return {
            "valid": False,
            "structure_issues": structure_issues,
            "images_issues": [],
            "label_issues": [],
            "stats": {},
            "class_counts": Counter(),
        }

    images_issues, stats = validate_images_and_labels(dataset_path)
    label_issues, class_counts = validate_label_format(dataset_path)

    total_issues = len(structure_issues) + len(images_issues) + len(label_issues)

    return {
        "valid": total_issues == 0,
        "structure_issues": structure_issues,
        "images_issues": images_issues,
        "label_issues": label_issues,
        "stats": stats,
        "class_counts": class_counts,
        "total_issues": total_issues,
    }
