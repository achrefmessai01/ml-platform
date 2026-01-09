from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from PIL import Image


class CropExtractor:
    """Extract crops from object detection datasets for classification."""

    def __init__(
        self,
        dataset_path: Union[str, Path],
        output_dir: Union[str, Path],
        crop_size: Optional[Tuple[int, int]] = None,
        padding: float = 0.1,
        min_crop_size: int = 32,
        preserve_splits: bool = True,
        quality: int = 95,
    ) -> None:
        """Initialize CropExtractor."""
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.crop_size = crop_size
        self.padding = padding
        self.min_crop_size = min_crop_size
        self.preserve_splits = preserve_splits
        self.quality = quality

        self.class_names = self._load_class_names()
        self.stats = {"total_crops": 0, "crops_per_class": {}}

        self._create_output_dirs()

    def _load_class_names(self) -> List[str]:
        """Load class names from dataset."""
        data_yaml_path = self.dataset_path / "data.yaml"
        if data_yaml_path.exists():
            with open(data_yaml_path, "r") as f:
                data = yaml.safe_load(f)
                if "names" in data:
                    if isinstance(data["names"], dict):
                        return [data["names"][i] for i in sorted(data["names"].keys())]
                    return data["names"]

        classes_path = self.dataset_path / "classes.txt"
        if classes_path.exists():
            with open(classes_path, "r") as f:
                return [line.strip() for line in f.readlines()]

        max_class = 0
        for split in ["train", "val", "test"]:
            labels_dir = self.dataset_path / split / "labels"
            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                max_class = max(max_class, class_id)

        return [f"class_{i}" for i in range(max_class + 1)]

    def _create_output_dirs(self) -> None:
        """Create output directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.preserve_splits:
            for split in ["train", "val", "test"]:
                split_dir = self.output_dir / split
                split_dir.mkdir(exist_ok=True)
                for class_name in self.class_names:
                    (split_dir / class_name).mkdir(exist_ok=True)
        else:
            for class_name in self.class_names:
                (self.output_dir / class_name).mkdir(exist_ok=True)

        for class_name in self.class_names:
            self.stats["crops_per_class"][class_name] = 0

    def _parse_annotation(self, line: str, img_width: int, img_height: int) -> Optional[Tuple[int, int, int, int, int]]:
        """Parse YOLO annotation and return class_id and bbox coordinates."""
        parts = line.strip().split()
        if len(parts) < 5:
            return None

        class_id = int(parts[0])

        if len(parts) == 5:  # AABB format
            x_center, y_center, width, height = map(float, parts[1:5])
            x1 = int((x_center - width / 2) * img_width)
            y1 = int((y_center - height / 2) * img_height)
            x2 = int((x_center + width / 2) * img_width)
            y2 = int((y_center + height / 2) * img_height)
            return class_id, x1, y1, x2, y2

        return None

    def _extract_crop(
        self, image: np.ndarray, bbox: Tuple[int, int, int, int], class_name: str
    ) -> Optional[np.ndarray]:
        """Extract and process crop from image."""
        x1, y1, x2, y2 = bbox

        # Apply padding
        width = x2 - x1
        height = y2 - y1

        pad_w = int(width * self.padding)
        pad_h = int(height * self.padding)

        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(image.shape[1], x2 + pad_w)
        y2 = min(image.shape[0], y2 + pad_h)

        crop = image[y1:y2, x1:x2]

        if crop.shape[0] < self.min_crop_size or crop.shape[1] < self.min_crop_size:
            return None

        if self.crop_size:
            crop = cv2.resize(crop, self.crop_size)

        return crop

    def _save_crop(self, crop: np.ndarray, output_path: Path) -> None:
        """Save crop image."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert BGR to RGB for PIL
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(crop_rgb)
        img.save(output_path, quality=self.quality, optimize=True)

    def extract_crops(self) -> None:
        """Extract crops from all images in the dataset."""
        splits = ["train", "val", "test"] if self.preserve_splits else ["."]

        for split in splits:
            if split == ".":
                images_dir = self.dataset_path / "images"
                labels_dir = self.dataset_path / "labels"
                output_base = self.output_dir
            else:
                images_dir = self.dataset_path / split / "images"
                labels_dir = self.dataset_path / split / "labels"
                output_base = self.output_dir / split if self.preserve_splits else self.output_dir

            if not images_dir.exists() or not labels_dir.exists():
                continue

            for img_path in images_dir.glob("*"):
                if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue

                label_path = labels_dir / f"{img_path.stem}.txt"
                if not label_path.exists():
                    continue

                self._extract_crops_from_image(img_path, label_path, output_base)

    def _extract_crops_from_image(self, img_path: Path, label_path: Path, output_base: Path) -> None:
        """Extract crops from a single image."""
        image = cv2.imread(str(img_path))
        if image is None:
            return

        img_height, img_width = image.shape[:2]

        with open(label_path, "r") as f:
            lines = f.readlines()

        for i, line in enumerate(lines):
            parsed = self._parse_annotation(line, img_width, img_height)
            if parsed is None:
                continue

            class_id, x1, y1, x2, y2 = parsed
            if class_id >= len(self.class_names):
                continue

            class_name = self.class_names[class_id]
            crop = self._extract_crop(image, (x1, y1, x2, y2), class_name)

            if crop is not None:
                crop_filename = f"{img_path.stem}_{i}.jpg"
                output_path = output_base / class_name / crop_filename
                self._save_crop(crop, output_path)

                self.stats["total_crops"] += 1
                self.stats["crops_per_class"][class_name] += 1

    def get_stats(self) -> Dict:
        """Get extraction statistics."""
        return self.stats.copy()
