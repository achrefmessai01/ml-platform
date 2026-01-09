#!/usr/bin/env python3
"""
Merge two COCO datasets by combining images, annotations, and categories.
The script handles category ID remapping when categories from both datasets are combined.
Optionally, all labels can be merged into a single class with id 0.
"""

import json
import shutil
from pathlib import Path
import argparse
from datetime import datetime


def load_coco_dataset(annotations_path):
    """Load COCO dataset from JSON file."""
    with open(annotations_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_coco_datasets(
    dataset1_path, dataset2_path, output_path, copy_images=True, single_class=False
):
    """
    Merge two COCO datasets into one.

    Args:
        dataset1_path: Path to first dataset directory
        dataset2_path: Path to second dataset directory
        output_path: Path to output merged dataset directory
        copy_images: Whether to copy image files (default: True)
        single_class: Whether to merge all labels into a single class with id 0 (default: False)
    """

    # Create output directory structure
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "annotations").mkdir(exist_ok=True)
    (output_path / "train").mkdir(exist_ok=True)
    (output_path / "val").mkdir(exist_ok=True)
    (output_path / "test").mkdir(exist_ok=True)

    # Process each split (train, val, test)
    for split in ["train", "val", "test"]:
        print(f"\nProcessing {split} split...")

        # Paths for annotation files
        ann1_path = Path(dataset1_path) / "annotations" / f"instances_{split}.json"
        ann2_path = Path(dataset2_path) / "annotations" / f"instances_{split}.json"

        # Skip if annotation files don't exist
        if not ann1_path.exists() or not ann2_path.exists():
            print(f"Skipping {split} - annotation files not found")
            continue

        # Load datasets
        dataset1 = load_coco_dataset(ann1_path)
        dataset2 = load_coco_dataset(ann2_path)

        # Create merged dataset structure
        merged_dataset = {
            "info": {
                "year": datetime.now().year,
                "version": "1.0",
                "description": f"Merged COCO dataset from {dataset1_path} and {dataset2_path}",
                "contributor": "Merge Script",
                "url": "",
                "date_created": datetime.now().isoformat(),
            },
            "licenses": dataset1.get("licenses", []) + dataset2.get("licenses", []),
            "categories": [],
            "images": [],
            "annotations": [],
        }

        # Combine categories and create mapping for dataset2
        category_mapping = {}  # old_id -> new_id for dataset2
        next_category_id = 0

        if single_class:
            # Create single category with id 0
            merged_dataset["categories"] = [
                {"id": 0, "name": "object", "supercategory": ""}
            ]
            # Map all categories to id 0
            for cat in dataset1.get("categories", []):
                category_mapping[cat["id"]] = 0
            for cat in dataset2.get("categories", []):
                category_mapping[cat["id"]] = 0
            print("Single class mode: All categories mapped to 'object' with id 0")
        else:
            # Add categories from dataset1
            for cat in dataset1.get("categories", []):
                merged_dataset["categories"].append(
                    {
                        "id": next_category_id,
                        "name": cat["name"],
                        "supercategory": cat.get("supercategory", ""),
                    }
                )
                next_category_id += 1

            # Add categories from dataset2 (avoiding duplicates)
            existing_names = {cat["name"] for cat in merged_dataset["categories"]}
            for cat in dataset2.get("categories", []):
                if cat["name"] not in existing_names:
                    # New category
                    new_id = next_category_id
                    merged_dataset["categories"].append(
                        {
                            "id": new_id,
                            "name": cat["name"],
                            "supercategory": cat.get("supercategory", ""),
                        }
                    )
                    category_mapping[cat["id"]] = new_id
                    next_category_id += 1
                else:
                    # Existing category - find its new ID
                    for existing_cat in merged_dataset["categories"]:
                        if existing_cat["name"] == cat["name"]:
                            category_mapping[cat["id"]] = existing_cat["id"]
                            break

        print(f"Categories: {len(merged_dataset['categories'])} total")
        for cat in merged_dataset["categories"]:
            print(f"  ID {cat['id']}: {cat['name']}")

        # Track image and annotation IDs
        next_image_id = 0
        next_annotation_id = 0
        image_id_mapping1 = {}  # old_id -> new_id for dataset1
        image_id_mapping2 = {}  # old_id -> new_id for dataset2

        # Add images and annotations from dataset1
        print(f"Adding {len(dataset1.get('images', []))} images from dataset1...")
        for img in dataset1.get("images", []):
            new_image_id = next_image_id
            image_id_mapping1[img["id"]] = new_image_id

            merged_dataset["images"].append(
                {
                    "id": new_image_id,
                    "width": img["width"],
                    "height": img["height"],
                    "file_name": f"dataset1_{img['file_name']}",  # Prefix to avoid conflicts
                    "license": img.get("license"),
                    "flickr_url": img.get("flickr_url"),
                    "coco_url": img.get("coco_url"),
                    "date_captured": img.get("date_captured"),
                }
            )
            next_image_id += 1

        # Add annotations from dataset1
        for ann in dataset1.get("annotations", []):
            merged_dataset["annotations"].append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id_mapping1[ann["image_id"]],
                    "category_id": 0
                    if single_class
                    else ann["category_id"],  # Use 0 if single class mode
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "bbox": ann["bbox"],
                    "iscrowd": ann["iscrowd"],
                }
            )
            next_annotation_id += 1

        # Add images and annotations from dataset2
        print(f"Adding {len(dataset2.get('images', []))} images from dataset2...")
        for img in dataset2.get("images", []):
            new_image_id = next_image_id
            image_id_mapping2[img["id"]] = new_image_id

            merged_dataset["images"].append(
                {
                    "id": new_image_id,
                    "width": img["width"],
                    "height": img["height"],
                    "file_name": f"dataset2_{img['file_name']}",  # Prefix to avoid conflicts
                    "license": img.get("license"),
                    "flickr_url": img.get("flickr_url"),
                    "coco_url": img.get("coco_url"),
                    "date_captured": img.get("date_captured"),
                }
            )
            next_image_id += 1

        # Add annotations from dataset2 with category remapping
        for ann in dataset2.get("annotations", []):
            merged_dataset["annotations"].append(
                {
                    "id": next_annotation_id,
                    "image_id": image_id_mapping2[ann["image_id"]],
                    "category_id": 0
                    if single_class
                    else category_mapping[
                        ann["category_id"]
                    ],  # Use 0 if single class mode
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "bbox": ann["bbox"],
                    "iscrowd": ann["iscrowd"],
                }
            )
            next_annotation_id += 1

        # Save merged annotations
        output_ann_path = output_path / "annotations" / f"instances_{split}.json"
        with open(output_ann_path, "w", encoding="utf-8") as f:
            json.dump(merged_dataset, f, indent=2)

        print(
            f"Saved merged {split} annotations: {len(merged_dataset['images'])} images, {len(merged_dataset['annotations'])} annotations"
        )

        # Copy images if requested
        if copy_images:
            print(f"Copying images for {split} split...")

            # Copy from dataset1
            dataset1_img_dir = Path(dataset1_path) / split
            if dataset1_img_dir.exists():
                for img in dataset1.get("images", []):
                    src_path = dataset1_img_dir / img["file_name"]
                    dst_path = output_path / split / f"dataset1_{img['file_name']}"
                    if src_path.exists():
                        shutil.copy2(src_path, dst_path)

            # Copy from dataset2
            dataset2_img_dir = Path(dataset2_path) / split
            if dataset2_img_dir.exists():
                for img in dataset2.get("images", []):
                    src_path = dataset2_img_dir / img["file_name"]
                    dst_path = output_path / split / f"dataset2_{img['file_name']}"
                    if src_path.exists():
                        shutil.copy2(src_path, dst_path)


def main():
    parser = argparse.ArgumentParser(description="Merge two COCO datasets")
    parser.add_argument("dataset1", help="Path to first COCO dataset directory")
    parser.add_argument("dataset2", help="Path to second COCO dataset directory")
    parser.add_argument("output", help="Path to output merged dataset directory")
    parser.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Don't copy image files, only merge annotations",
    )
    parser.add_argument(
        "--single-class",
        action="store_true",
        help="Merge all labels into a single class with id 0",
    )

    args = parser.parse_args()

    print("Merging datasets:")
    print(f"  Dataset 1: {args.dataset1}")
    print(f"  Dataset 2: {args.dataset2}")
    print(f"  Output: {args.output}")
    print(f"  Copy images: {not args.no_copy_images}")
    print(f"  Single class mode: {args.single_class}")

    merge_coco_datasets(
        args.dataset1,
        args.dataset2,
        args.output,
        copy_images=not args.no_copy_images,
        single_class=args.single_class,
    )

    print(f"\nMerge completed! Merged dataset saved to: {args.output}")
    if args.single_class:
        print("All annotations have been mapped to a single 'object' class with id 0.")


if __name__ == "__main__":
    main()
