from pathlib import Path
from typing import List, Optional, Union
from ultralytics import YOLO
import cv2
import numpy as np
import torch


class StrawberryDetector:
    """YOLO11-based strawberry detector"""
    
    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
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
        iou: float = 0.45,
        imgsz: int = 640,
        save: bool = False,
        project: str = "runs/predict",
        name: str = "exp"
    ):
        """Run detection on images"""
        return self.model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=self.device,
            save=save,
            project=project,
            name=name,
            verbose=False
        )
    
    def train(
        self,
        data: Union[str, Path],
        epochs: int = 100,
        batch: int = 16,
        imgsz: int = 640,
        project: str = "runs/train",
        name: str = "exp",
        **kwargs
    ):
        """Train the detection model"""
        return self.model.train(
            data=str(data),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            project=project,
            name=name,
            verbose=False,
            **kwargs
        )
    
    def validate(
        self,
        data: Union[str, Path],
        batch: int = 16,
        imgsz: int = 640,
        project: str = "runs/val",
        name: str = "exp"
    ):
        """Validate the detection model"""
        return self.model.val(
            data=str(data),
            batch=batch,
            imgsz=imgsz,
            device=self.device,
            project=project,
            name=name,
            verbose=False
        )
