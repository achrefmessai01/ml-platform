#!/usr/bin/env python3
"""
Clean MLflow + Detectron2 inference script.
Loads model weights from MLflow and runs strawberry detection.
"""

import os
import shutil
import tempfile

import cv2
import mlflow
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from mlflow.tracking import MlflowClient


def load_model_from_mlflow(
    model_name, model_version="latest", tracking_uri="https://gitlab.com/api/v4/projects/72607084/ml/mlflow/"
):
    """Download model weights from MLflow run artifacts."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get model version and run ID
    if model_version == "latest":
        model_versions = client.get_latest_versions(model_name, stages=["None"])
        if not model_versions:
            model_versions = client.get_latest_versions(
                model_name, stages=["Production", "Staging"]
            )
        if not model_versions:
            raise ValueError(f"No versions found for model {model_name}")
        model_version = model_versions[0]
    else:
        model_version = client.get_model_version(model_name, model_version)

    run_id = model_version.run_id
    print(f"Loading model from run: {run_id}")

    # Download artifacts
    temp_dir = tempfile.mkdtemp()
    try:
        downloaded_path = client.download_artifacts(run_id, "", temp_dir)

        # Find .pth files
        pth_files = []
        for root, dirs, files in os.walk(downloaded_path):
            for file in files:
                if file.endswith(".pth"):
                    pth_files.append(os.path.join(root, file))

        if not pth_files:
            raise FileNotFoundError("No .pth model files found in MLflow artifacts")

        # Use the first .pth file found
        model_weights_path = pth_files[0]
        print(f"Found model weights: {os.path.basename(model_weights_path)}")

        return model_weights_path, temp_dir

    except Exception as e:
        shutil.rmtree(temp_dir)
        raise e


def main():
    # Configuration
    MODEL_NAME = "strawberry-detector-faster-rcnn"
    MODEL_VERSION = "latest"
    TRACKING_URI = "https://gitlab.com/api/v4/projects/72607084/ml/mlflow/"
    IMAGE_PATH = "datasets/merged_data/detection/combined_coco/test/dataset1_05d2f321-IMG_9823.JPEG"
    OUTPUT_PATH = "detection_result_mlflow.jpg"

    # Load image
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        print(f"Error: Could not load image from {IMAGE_PATH}")
        return
    print(f"Loaded image: {IMAGE_PATH}")

    # Setup Detectron2 config
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # strawberry class
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # Load model weights
    temp_dir = None
    try:
        model_weights_path, temp_dir = load_model_from_mlflow(
            MODEL_NAME, MODEL_VERSION, TRACKING_URI
        )
        cfg.MODEL.WEIGHTS = model_weights_path
        print("Successfully loaded MLflow model weights")

    except Exception as e:
        print(f"Could not load MLflow model: {e}")
        print("Falling back to pretrained COCO weights")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
        )
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 80  # COCO classes

    # Create predictor and run inference
    predictor = DefaultPredictor(cfg)
    print("Running inference...")
    outputs = predictor(image)

    # Clean up temporary files
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    # Process results
    instances = outputs["instances"].to("cpu")
    print(f"Detected {len(instances)} objects")

    if len(instances) > 0:
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        print(f"Confidence scores: {scores}")
        print(f"Predicted classes: {classes}")

    # Visualize results
    metadata = MetadataCatalog.get("strawberry_inference")
    metadata.thing_classes = (
        ["strawberry"] if cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1 else ["object"]
    )

    v = Visualizer(image[:, :, ::-1], metadata=metadata, scale=1.0)
    vis = v.draw_instance_predictions(instances)
    result_image = vis.get_image()[:, :, ::-1]

    # Save result
    cv2.imwrite(OUTPUT_PATH, result_image)
    print(f"Detection visualization saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
