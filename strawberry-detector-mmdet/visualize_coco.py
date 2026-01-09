#!/usr/bin/env python3

"""
Visualize COCO dataset samples with actual bounding box annotations using MMCV/MMDetection.
Loads real annotations from the dataset and displays them on the images.
"""

import argparse
import mmcv
import numpy as np
import os
from mmengine.config import Config
from mmengine.registry import build_from_cfg
from mmdet.registry import DATASETS


def show_sample(dataset, idx, out_file=None):
    """Visualize a sample from the dataset with real annotations"""

    # Get data info
    data_info = dataset.get_data_info(idx)
    img_path = data_info["img_path"]

    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return False

    # Load image
    img = mmcv.imread(img_path)
    print(f"Image loaded: {img_path}")
    print(f"Image shape: {img.shape}")

    # Get annotations
    instances = data_info.get("instances", [])

    if len(instances) > 0:
        bboxes = np.array([inst["bbox"] for inst in instances], dtype=np.float32)
        labels = np.array([inst["bbox_label"] for inst in instances], dtype=np.int64)
        print(f"Found {len(bboxes)} annotations")
    else:
        print("No annotations found for this image")
        return False

    # Get class names
    class_names = getattr(dataset, "metainfo", {}).get("classes", ["unknown"])

    if out_file is None:
        out_file = f"sample_{idx}_visualization.jpg"

    # Visualize with bounding boxes
    mmcv.imshow_det_bboxes(
        img,
        bboxes,
        labels,
        class_names=class_names,
        show=False,  # Don't display (headless environment)
        bbox_color="green",
        text_color="white",
        thickness=2,
        out_file=out_file,
    )

    print(f"✓ Saved visualization to: {out_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize COCO dataset samples")
    parser.add_argument("config", help="Config file path")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to visualize",
    )
    parser.add_argument("--idx", type=int, default=0, help="Sample index to visualize")
    parser.add_argument("--out-file", help="Output file path for visualization")
    args = parser.parse_args()

    # Load config
    try:
        cfg = Config.fromfile(args.config)
        print(f"✓ Loaded config: {args.config}")
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return

    # Get dataset config
    dataset_cfg = cfg.data[args.split]

    # Build dataset
    try:
        dataset = build_from_cfg(dataset_cfg, DATASETS)
        print(f"✓ Built dataset with {len(dataset)} samples")

        class_names = getattr(dataset, "metainfo", {}).get("classes", ["unknown"])
        print(f"✓ Classes: {class_names}")

    except Exception as e:
        print(f"✗ Failed to build dataset: {e}")
        return

    # Visualize sample
    success = show_sample(dataset, args.idx, args.out_file)

    if success:
        print(f"✓ Successfully visualized sample {args.idx}")
    else:
        print(f"✗ Failed to visualize sample {args.idx}")


if __name__ == "__main__":
    main()
