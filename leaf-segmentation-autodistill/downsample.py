#!/usr/bin/env python3
"""
Clean and minimal script to downscale images in Label Studio format.

Resizes images to not exceed maximum resolution and updates label coordinates proportionally.
"""

import argparse
import json
from pathlib import Path

from PIL import Image


def get_scale_factor(img_width: int, img_height: int, max_resolution: int) -> float:
    """Calculate scale factor to ensure no dimension exceeds max_resolution."""
    max_dimension = max(img_width, img_height)
    if max_dimension <= max_resolution:
        return 1.0
    return max_resolution / max_dimension


def scale_coordinates(coordinates: list[list[float]], scale_factor: float) -> list[list[float]]:
    """Scale coordinate points by the given factor."""
    return [[x * scale_factor, y * scale_factor] for x, y in coordinates]


def process_image(image_path: Path, output_path: Path, max_resolution: int) -> tuple[float, int, int]:
    """Resize image if needed and return scale factor."""
    with Image.open(image_path) as img:
        original_width, original_height = img.size
        scale_factor = get_scale_factor(original_width, original_height, max_resolution)

        if scale_factor < 1.0:
            new_width = int(original_width * scale_factor)
            new_height = int(original_height * scale_factor)
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(output_path)
            print(f"Resized {image_path.name}: {original_width}x{original_height} -> {new_width}x{new_height}")
        else:
            # Copy original if no resize needed
            img.save(output_path)
            print(f"Copied {image_path.name}: {original_width}x{original_height} (no resize needed)")

        return scale_factor, original_width, original_height


def update_label_file(
    label_path: Path, output_label_path: Path, scale_factor: float, new_width: int, new_height: int
) -> None:
    """Update label file with scaled coordinates and new image dimensions."""
    with open(label_path, "r") as f:
        label_data = json.load(f)

    # Update annotations
    for annotation in label_data.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "polygonlabels":
                # Update image dimensions
                result["original_width"] = new_width
                result["original_height"] = new_height

                # Scale polygon points
                if "value" in result and "points" in result["value"]:
                    result["value"]["points"] = scale_coordinates(result["value"]["points"], scale_factor)

    # Save updated label file
    with open(output_label_path, "w") as f:
        json.dump(label_data, f, indent=2)


def main() -> None:
    """Main function to parse arguments and process dataset."""
    parser = argparse.ArgumentParser(description="Downscale Label Studio dataset images and update labels")
    parser.add_argument("dataset_path", type=Path, help="Path to Label Studio dataset directory")
    parser.add_argument(
        "--max-resolution", type=int, default=1024, help="Maximum resolution for any image dimension (default: 1024)"
    )
    parser.add_argument("--output-dir", type=Path, help='Output directory (default: dataset_path + "_downscaled")')

    args = parser.parse_args()

    # Setup paths
    dataset_path = args.dataset_path
    output_dir = args.output_dir or Path(f"{dataset_path}_downscaled")

    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"

    output_images_dir = output_dir / "images"
    output_labels_dir = output_dir / "labels"

    # Validate input
    if not dataset_path.exists():
        print(f"Error: Dataset path '{dataset_path}' does not exist")
        return

    if not images_dir.exists() or not labels_dir.exists():
        print(f"Error: Expected 'images' and 'labels' directories in '{dataset_path}'")
        return

    # Create output directories
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Copy data.yaml if it exists
    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
        import shutil

        shutil.copy2(data_yaml, output_dir / "data.yaml")

    # Process images and labels
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    processed_count = 0

    for image_path in image_files:
        # Find corresponding label file
        label_name = image_path.stem + ".json"
        label_path = labels_dir / label_name

        if not label_path.exists():
            print(f"Warning: No label file found for {image_path.name}")
            continue

        # Process image
        output_image_path = output_images_dir / image_path.name
        output_label_path = output_labels_dir / label_name

        scale_factor, orig_width, orig_height = process_image(image_path, output_image_path, args.max_resolution)

        # Calculate new dimensions
        new_width = int(orig_width * scale_factor)
        new_height = int(orig_height * scale_factor)

        # Update label file
        update_label_file(label_path, output_label_path, scale_factor, new_width, new_height)

        processed_count += 1

    print(f"\nProcessed {processed_count} image-label pairs")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
