"""Strawberry Classifier - YOLO11-based strawberry detection and classification."""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .core.classifier import StrawberryClassifier
from .core.detector import StrawberryDetector
from .data import (
    CropExtractor,
    analyze_dataset,
    apply_color_transfer_to_dataset,
    compare_datasets,
    create_subset,
    merge_datasets,
    prepare_dataset,
    validate_dataset,
)

__all__ = [
    "StrawberryDetector",
    "StrawberryClassifier",
    "prepare_dataset",
    "CropExtractor",
    "apply_color_transfer_to_dataset",
    "compare_datasets",
    "analyze_dataset",
    "validate_dataset",
    "merge_datasets",
    "create_subset",
]
