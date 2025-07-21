import shutil
#!/usr/bin/env python3
"""
Convert YOLO segmentation labels (polygon) to YOLO Oriented Bounding Box (OBB) labels.
Input: YOLO segmentation label files (class_id x1 y1 x2 y2 ... xn yn, normalized)
Output: YOLO OBB label files (class_id cx cy w h angle, normalized, angle in degrees)

Requires: opencv-python, numpy
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2


def parse_args():
    parser = argparse.ArgumentParser(description="Convert YOLO segmentation labels to YOLO OBB labels.")
    parser.add_argument('--input_dataset', help='Path to input dataset directory (should contain labels/ and images/)')
    parser.add_argument('--output_dataset', help='Path to output dataset directory (labels/ will be created here)')
    return parser.parse_args()


def get_image_size(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img.shape[1], img.shape[0]  # width, height


def polygon_to_obb(points):
    # points: Nx2 array, absolute pixel coordinates
    rect = cv2.minAreaRect(points.astype(np.float32))
    box = cv2.boxPoints(rect)  # 4x2 array of corner points (float32)
    return box  # shape (4,2)


def process_label_file(label_path, img_path, output_path):
    width, height = get_image_size(img_path)
    obb_lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue  # not a polygon
            class_id = parts[0]
            coords = np.array(list(map(float, parts[1:])), dtype=np.float32)
            xs = np.array(coords[0::2]) * width
            ys = np.array(coords[1::2]) * height
            points = np.column_stack((xs, ys))
            box = polygon_to_obb(points)  # shape (4,2)
            # Normalize box points
            box_n = np.zeros_like(box, dtype=box.dtype)
            box_n[:, 0] = box[:, 0] / width
            box_n[:, 1] = box[:, 1] / height
            # Flatten to x1 y1 x2 y2 x3 y3 x4 y4
            box_flat = box_n.flatten()
            obb_line = f"{class_id} " + " ".join(f"{v:.6f}" for v in box_flat)
            obb_lines.append(obb_line)
    if obb_lines:
        with open(output_path, 'w') as f:
            f.write('\n'.join(obb_lines) + '\n')
def cleanup_generated_labels(output_dir):
    # Remove all .txt files in the output_dir
    files = glob.glob(os.path.join(output_dir, '*.txt'))
    for fpath in files:
        try:
            os.remove(fpath)
        except Exception as e:
            print(f"Failed to remove {fpath}: {e}")


def main():
    args = parse_args()
    splits = ['train', 'val', 'test', 'valid']
    # Ensure output root directory exists
    os.makedirs(args.output_dataset, exist_ok=True)
    # Copy data.yaml if it exists
    data_yaml_in = os.path.join(args.input_dataset, 'data.yaml')
    data_yaml_out = os.path.join(args.output_dataset, 'data.yaml')
    if os.path.isfile(data_yaml_in):
        shutil.copy2(data_yaml_in, data_yaml_out)
        print(f"Copied: {data_yaml_in} -> {data_yaml_out}")

    for split in splits:
        input_split = os.path.join(args.input_dataset, split)
        if not os.path.isdir(input_split):
            continue
        input_labels = os.path.join(input_split, 'labels')
        input_images = os.path.join(input_split, 'images')
        output_split = os.path.join(args.output_dataset, split)
        output_labels = os.path.join(output_split, 'labels')
        output_images = os.path.join(output_split, 'images')
        os.makedirs(output_labels, exist_ok=True)
        os.makedirs(output_images, exist_ok=True)
        # Clean up previously generated OBB label files
        cleanup_generated_labels(output_labels)

        # Try to infer image extension from the first image in the input_images dir
        img_ext = '.jpg'
        img_files = glob.glob(os.path.join(input_images, '*'))
        if img_files:
            img_ext = os.path.splitext(img_files[0])[1]

        label_files = glob.glob(os.path.join(input_labels, '*.txt'))
        for label_path in label_files:
            base = os.path.splitext(os.path.basename(label_path))[0]
            img_path = os.path.join(input_images, base + img_ext)
            output_path = os.path.join(output_labels, base + '.txt')
            try:
                process_label_file(label_path, img_path, output_path)
                print(f"Converted: {label_path} -> {output_path}")
            except Exception as e:
                print(f"Error processing {label_path}: {e}")

        # Copy all images for this split
        for img_file in glob.glob(os.path.join(input_images, '*')):
            out_img_file = os.path.join(output_images, os.path.basename(img_file))
            try:
                shutil.copy2(img_file, out_img_file)
            except Exception as e:
                print(f"Failed to copy image {img_file} to {out_img_file}: {e}")

if __name__ == "__main__":
    main()
