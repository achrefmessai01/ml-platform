import random

import cv2
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer

register_coco_instances(
    "strawberries_train",
    {},
    "datasets/merged_data/detection/combined_coco/annotations/instances_train.json",
    "datasets/merged_data/detection/combined_coco/train",
)
register_coco_instances(
    "strawberries_val",
    {},
    "datasets/merged_data/detection/combined_coco/annotations/instances_val.json",
    "datasets/merged_data/detection/combined_coco/val",
)


# Visualize a random sample from the training dataset
def visualize_random_sample() -> None:
    """Visualize a random sample from the strawberries training dataset."""
    dataset_dicts = DatasetCatalog.get("strawberries_train")
    dataset_metadata = MetadataCatalog.get("strawberries_train")

    # Select a random sample
    sample = random.choice(dataset_dicts)

    # Read the image
    img = cv2.imread(sample["file_name"])

    # Create visualizer
    visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=0.8)

    # Add annotations
    vis = visualizer.draw_dataset_dict(sample)

    # Save the image
    output_path = "visualized_sample.jpg"
    # Convert RGB back to BGR for cv2.imwrite
    cv2.imwrite(output_path, vis.get_image()[:, :, ::-1])
    print(f"Visualized sample saved to: {output_path}")
    print(f"Original image: {sample['file_name']}")


if __name__ == "__main__":
    visualize_random_sample()
