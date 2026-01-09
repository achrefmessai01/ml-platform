import glob
import json
import os

import cv2
import numpy as np


def mask_to_rle(mask: np.ndarray) -> list:
    """
    Convert a binary mask to RLE format for Label Studio.

    Args:
        mask (numpy.ndarray): Binary mask (0s and 1s)

    Returns:
        list: RLE encoded mask
    """
    # Flatten the mask
    flat_mask = mask.flatten()

    # Convert to RLE
    rle = []
    current_value = flat_mask[0]
    count = 1

    for i in range(1, len(flat_mask)):
        if flat_mask[i] == current_value:
            count += 1
        else:
            rle.extend([count, current_value])
            current_value = flat_mask[i]
            count = 1

    # Don't forget the last run
    rle.extend([count, current_value])

    # Convert numpy types to Python native types for JSON serialization
    return [int(x) for x in rle]


def polygons2masks(polygons: list[list[float]], image_size: tuple[int, int]) -> np.ndarray:
    """
    Convert a list of polygons to a single binary mask for the whole image.

    Args:
        polygons (list): A list of polygons, where each polygon is a list of [x, y] coordinates.
        image_size (tuple): A (width, height) tuple for the mask dimensions.
    """
    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    for polygon in polygons:
        # Convert polygon points to np.int32 format, as cv2.fillPoly requires it
        points = np.array(polygon, dtype=np.int32)

        # cv2.fillPoly expects a list of contours
        cv2.fillPoly(mask, [points], (255,))  # Draw polygon with value 255

    return mask


def read_ls_polygons_and_size(json_path: str) -> tuple[list[list[float]], tuple[int, int]]:
    """
    Read polygons and image size from a Label Studio JSON task file.

    Converts percentage coordinates from Label Studio to absolute pixel coordinates.
    """
    polygons = []
    image_size = (0, 0)  # (width, height)

    with open(json_path, "r") as f:
        data = json.load(f)

    # Find the image dimensions.
    # We assume all annotations in a file share the same dimensions.
    width, height = None, None
    if data.get("annotations"):
        for result in data["annotations"][0].get("result", []):
            if "original_width" in result and "original_height" in result:
                width = result["original_width"]
                height = result["original_height"]
                image_size = (width, height)
                break

    if not width or not height:
        if "data" in data and "image" in data["data"]:
            print(f"Warning: Could not determine image size for {json_path}. Skipping.")
            return [], (0, 0)

    # Iterate through all annotations and results to find polygons
    for annotation in data.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "polygonlabels" and "value" in result:
                value = result["value"]

                if "points" in value:
                    points_percent = value["points"]

                    # Convert percentage points [x_perc, y_perc] to absolute [x_pix, y_pix]
                    abs_polygon = []
                    for x_perc, y_perc in points_percent:
                        abs_x = (x_perc / 100.0) * width
                        abs_y = (y_perc / 100.0) * height
                        abs_polygon.append([abs_x, abs_y])

                    polygons.append(abs_polygon)

    return polygons, image_size


def create_brush_label_task(original_data: dict, mask: np.ndarray, image_size: tuple[int, int]) -> dict:
    """
    Create a new Label Studio task with a single brushLabel combining all polygons.

    Args:
        original_data (dict): Original Label Studio task data
        mask (numpy.ndarray): Binary mask combining all polygons
        image_size (tuple): (width, height)

    Returns:
        dict: New Label Studio task with single combined brushLabel
    """
    width, height = image_size

    # Convert mask to binary (0s and 1s)
    binary_mask = (mask > 0).astype(np.uint8)

    # Convert to RLE
    rle = mask_to_rle(binary_mask)

    # Create new task structure
    new_task = {"data": original_data.get("data", {}), "annotations": []}

    # Collect all unique labels from all polygon annotations
    all_labels = set()
    first_polygon_result = None

    # Process original annotations to collect labels and metadata
    for annotation in original_data.get("annotations", []):
        for result in annotation.get("result", []):
            if result.get("type") == "polygonlabels":
                if first_polygon_result is None:
                    first_polygon_result = result
                labels = result["value"].get("polygonlabels", [])
                all_labels.update(labels)

    # Create a single annotation with one combined brush label
    if first_polygon_result and all_labels:
        for annotation in original_data.get("annotations", []):
            new_annotation = {
                "id": annotation.get("id"),
                "created_username": annotation.get("created_username"),
                "created_ago": annotation.get("created_ago"),
                "result": [],
            }

            # Create single brush label combining all polygons
            brush_result = {
                "original_width": first_polygon_result.get("original_width"),
                "original_height": first_polygon_result.get("original_height"),
                "image_rotation": first_polygon_result.get("image_rotation", 0),
                "value": {"format": "rle", "rle": rle, "brushlabels": list(all_labels)},
                "id": first_polygon_result.get("id"),
                "from_name": first_polygon_result.get("from_name"),
                "to_name": first_polygon_result.get("to_name"),
                "type": "brushlabels",
                "origin": "manual",
            }
            new_annotation["result"].append(brush_result)
            new_task["annotations"].append(new_annotation)
            break  # Only create one annotation

    return new_task


if __name__ == "__main__":
    labels_path = "datasets/merged_data/test"

    # Define the parallel output directory
    parent_dir = os.path.dirname(labels_path.rstrip("/"))
    output_path = labels_path + "_brush"

    # Create the output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    print(f"Saving brush label tasks to: {output_path}")

    # Find all JSON files in the labels directory
    json_files = glob.glob(os.path.join(labels_path, "*.json"))

    if not json_files:
        print(f"Warning: No .json files found in {labels_path}")

    processed_count = 0
    # Loop over each JSON file
    for json_file in json_files:
        try:
            # Read the original Label Studio task
            with open(json_file, "r") as f:
                original_data = json.load(f)

            # Read polygons and image size from the Label Studio JSON
            polygons, image_size = read_ls_polygons_and_size(json_file)

            if not polygons:
                print(f"No polygons found in {json_file}, skipping.")
                continue

            if image_size == (0, 0):
                print(f"Error: Could not get image size for {json_file}, skipping.")
                continue

            # Convert all polygons to a single binary mask
            mask = polygons2masks(polygons, image_size)

            # Create new Label Studio task with brush labels
            brush_task = create_brush_label_task(original_data, mask, image_size)

            # Define the output save path
            base_name = os.path.basename(json_file)
            save_path = os.path.join(output_path, base_name)

            # Save the new task as JSON
            with open(save_path, "w") as f:
                json.dump(brush_task, f, indent=2)

            processed_count += 1

        except Exception as e:
            print(f"Error processing {json_file}: {e}")

    print(f"\nDone! Processed {processed_count} files.")
