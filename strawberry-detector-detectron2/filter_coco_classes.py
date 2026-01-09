#!/usr/bin/env python3
"""
Simple COCO Class Filter.

Filters classes from COCO datasets by removing unwanted classes and their bounding boxes.
Creates a new folder with modified annotations only (no image copying).

Usage:
    python filter_coco_classes.py input_dataset output_dataset class1,class2,class3
"""

import json
import sys
from pathlib import Path


def filter_coco_dataset(dataset_dir: str, classes_to_remove: list[str]) -> None:
    """Filter COCO dataset by removing specified classes."""
    dataset_path = Path(dataset_dir)

    # Check input directory
    annotations_dir = dataset_path / "annotations"
    if not annotations_dir.exists():
        print(f"Error: {annotations_dir} not found")
        return

    # Create output annotations directory
    output_annotations_dir = dataset_path / "annotations_filtered"
    output_annotations_dir.mkdir(parents=True, exist_ok=True)

    # Process each annotation file
    for ann_file in [
        "instances_train.json",
        "instances_val.json",
        "instances_test.json",
    ]:
        input_file = annotations_dir / ann_file
        output_file = output_annotations_dir / ann_file

        if not input_file.exists():
            continue

        print(f"Processing {ann_file}...")

        # Load annotations
        with open(input_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Filter categories - keep only classes not in removal list
        kept_categories = []
        valid_category_ids = set()
        category_mapping = {}

        for category in data["categories"]:
            if category["name"] not in classes_to_remove:
                new_category = category.copy()
                new_category["id"] = len(kept_categories)
                category_mapping[category["id"]] = new_category["id"]
                kept_categories.append(new_category)
                valid_category_ids.add(category["id"])

        # Filter annotations - keep only those with valid categories
        kept_annotations = []
        valid_image_ids = set()

        for annotation in data["annotations"]:
            if annotation["category_id"] in valid_category_ids:
                new_annotation = annotation.copy()
                new_annotation["category_id"] = category_mapping[annotation["category_id"]]
                kept_annotations.append(new_annotation)
                valid_image_ids.add(annotation["image_id"])

        # Filter images - keep only those with annotations
        kept_images = [img for img in data["images"] if img["id"] in valid_image_ids]

        # Create filtered dataset
        filtered_data = {
            "info": data.get("info", {}),
            "licenses": data.get("licenses", []),
            "categories": kept_categories,
            "images": kept_images,
            "annotations": kept_annotations,
        }

        # Save filtered annotations
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, indent=2)

        print(f"  Kept {len(kept_categories)}/{len(data['categories'])} classes")
        print(f"  Kept {len(kept_annotations)}/{len(data['annotations'])} annotations")
        print(f"  Kept {len(kept_images)}/{len(data['images'])} images")

    print(f"\nFiltered annotations saved to: {output_annotations_dir}")


def main() -> None:
    """Main function to parse arguments and call the filter function."""
    if len(sys.argv) != 3:
        print("Usage: python filter_coco_classes.py <input_dataset> <classes_to_remove>")
        print("Example: python filter_coco_classes.py coco_organized 'UNSURE,Flower'")
        sys.exit(1)

    dataset = sys.argv[1]
    classes_to_remove = [cls.strip() for cls in sys.argv[2].split(",")]

    print(f"Removing classes: {classes_to_remove}")
    filter_coco_dataset(dataset, classes_to_remove)


if __name__ == "__main__":
    main()
