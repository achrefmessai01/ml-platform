#!/usr/bin/env python3

"""
Fix image paths in COCO annotation files to match actual image locations.
Handles cases where annotation paths contain outdated relative paths or directory structures.
"""

import json
import os
import argparse
from pathlib import Path


def fix_annotation_paths(ann_file, images_dir, output_file=None, dry_run=False):
    """
    Fix image paths in COCO annotation file to match actual image locations.

    Args:
        ann_file: Path to COCO annotation file
        images_dir: Directory containing the actual images
        output_file: Output annotation file (overwrites input if None)
        dry_run: If True, only show what would be changed
    """

    # Load annotations
    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    # Get list of actual image files
    image_files = {}
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        for img_path in Path(images_dir).glob(ext):
            image_files[img_path.name.lower()] = img_path.name

    print(f"Found {len(image_files)} images in {images_dir}")
    print(f"Processing {len(coco_data['images'])} image entries in annotations")

    # Fix image paths
    fixed_count = 0
    missing_count = 0

    for img_info in coco_data["images"]:
        original_path = img_info["file_name"]

        # Extract just the filename from the path (remove directories)
        original_filename = Path(original_path).name
        filename_lower = original_filename.lower()

        if filename_lower in image_files:
            correct_filename = image_files[filename_lower]
            if original_path != correct_filename:
                if not dry_run:
                    img_info["file_name"] = correct_filename
                print(f"Fixed: {original_path} -> {correct_filename}")
                fixed_count += 1
            # else: path is already correct
        else:
            print(f"Missing: {original_path} (filename: {original_filename})")
            missing_count += 1

    print("\nSummary:")
    print(f"  Fixed paths: {fixed_count}")
    print(f"  Missing images: {missing_count}")
    print(
        f"  Unchanged entries: {len(coco_data['images']) - fixed_count - missing_count}"
    )
    print(f"  Total entries: {len(coco_data['images'])}")

    # Save updated annotations
    if not dry_run and fixed_count > 0:
        output_path = output_file or ann_file
        with open(output_path, "w") as f:
            json.dump(coco_data, f, indent=2)
        print(f"✓ Saved updated annotations to: {output_path}")
    elif dry_run:
        print("✓ Dry run completed - no files were modified")
    else:
        print("✓ No changes needed")


def main():
    parser = argparse.ArgumentParser(
        description="Fix image paths in COCO annotation files",
        epilog="""
Examples:
  # Preview changes without modifying files
  python fix_coco_paths.py annotations.json images/ --dry-run

  # Fix paths and overwrite original file
  python fix_coco_paths.py annotations.json images/

  # Fix paths and save to new file
  python fix_coco_paths.py annotations.json images/ --output fixed_annotations.json
        """,
    )
    parser.add_argument("annotation_file", help="COCO annotation file to fix")
    parser.add_argument("images_dir", help="Directory containing actual images")
    parser.add_argument(
        "--output", help="Output annotation file (default: overwrite input)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show changes without modifying files"
    )

    args = parser.parse_args()

    if not os.path.exists(args.annotation_file):
        print(f"Error: Annotation file not found: {args.annotation_file}")
        return 1

    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1

    fix_annotation_paths(
        args.annotation_file, args.images_dir, args.output, args.dry_run
    )

    return 0


if __name__ == "__main__":
    exit(main())
