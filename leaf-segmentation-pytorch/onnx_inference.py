"""Minimal ONNX inference script for leaf segmentation.

This script demonstrates how to use the exported ONNX model for inference.
The ONNX model includes all preprocessing (resizing, normalization) and
postprocessing (sigmoid, upsampling), so you only need to load an image and run inference.
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
from torchvision.io import read_image


def main() -> None:
    """Run ONNX inference on a leaf image."""
    # Configuration
    ONNX_MODEL_PATH = "leaf_segmentation.onnx"
    IMAGE_PATH = "datasets/strawberry/valid/dataset1_2bd8ab67-IMG_9344_jpg.rf.7d67c57277027a9cd34451062fdf2fe6.jpg"

    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]

    print(f"Loading ONNX model from {ONNX_MODEL_PATH}...")
    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_PATH)

    print(f"Loading image from {IMAGE_PATH}...")
    # Load image - NO preprocessing needed!
    image = read_image(IMAGE_PATH).float().numpy()  # (C, H, W) in [0, 255]

    # Add batch dimension: (C, H, W) -> (1, C, H, W)
    onnx_input = np.expand_dims(image, axis=0)

    print(f"Running inference on image with shape {onnx_input.shape}...")
    # Run inference - model handles all preprocessing and postprocessing internally
    ort_inputs = {ort_session.get_inputs()[0].name: onnx_input}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # Squeeze to remove batch and channel dimensions: (1, 1, H, W) -> (H, W)
    prediction = ort_output.squeeze()

    print(f"Output shape: {ort_output.shape}")
    print(f"Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")
    print(f"Output size matches input: {ort_output.shape[2:] == onnx_input.shape[2:]}")

    # Visualize results
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    # Convert from (C, H, W) to (H, W, C) for display and normalize to [0, 1]
    display_image = np.transpose(image, (1, 2, 0)) / 255.0
    plt.imshow(display_image)
    plt.title(f"Input Image ({image.shape[1]}x{image.shape[2]})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(prediction, cmap="gray", vmin=0, vmax=1)
    plt.title(f"Prediction ({prediction.shape[0]}x{prediction.shape[1]})")
    plt.axis("off")

    plt.tight_layout()

    # Save the result
    output_path = "onnx_inference_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Result saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
