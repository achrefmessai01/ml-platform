from pathlib import Path
from typing import Union

import numpy as np
import torch
from ultralytics import YOLO


class StrawberryClassifier:
    """YOLO11-based strawberry classifier."""

    def __init__(self, model_path: Union[str, Path], device: str = "auto") -> None:
        """Initialize StrawberryClassifier."""
        self.model_path = Path(model_path)
        # Auto-detect device: use CUDA if available, otherwise CPU
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = YOLO(str(self.model_path))

    def predict(
        self,
        source: Union[str, Path, np.ndarray],
        conf: float = 0.25,
        imgsz: int = 640,
        save: bool = False,
        project: str = "runs/classify",
        name: str = "exp",
    ) -> dict:
        """Run classification on images."""
        return self.model.predict(
            source=source,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            save=save,
            project=project,
            name=name,
            verbose=False,
        )

    def train(
        self,
        data: Union[str, Path],
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        project: str = "runs/train",
        name: str = "exp",
        **kwargs,
    ) -> dict:
        """Train the classification model."""
        return self.model.train(
            data=str(data),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            project=project,
            name=name,
            verbose=False,
            **kwargs,
        )

    def validate(
        self,
        data: Union[str, Path],
        batch: int = 16,
        imgsz: int = 640,
        project: str = "runs/val",
        name: str = "exp",
    ) -> dict:
        """Validate the classification model."""
        return self.model.val(
            data=str(data),
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            project=project,
            name=name,
            verbose=False,
        )
