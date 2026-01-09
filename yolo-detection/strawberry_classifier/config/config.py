from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..data.utils import load_yaml, save_yaml


@dataclass
class TrainingConfig:
    """Training configuration."""

    model: str = "yolo11n.pt"
    data: str = ""
    task: str = "detect"  # detect or classify
    epochs: int = 100
    batch_size: int = 16
    imgsz: int = 640
    device: str = "auto"
    project: str = "runs/train"
    name: str = "exp"
    patience: int = 10

    @classmethod
    def from_yaml(cls, config_path: Path) -> "TrainingConfig":
        """Load config from YAML file."""
        config_dict = load_yaml(config_path)
        return cls(**config_dict)

    def to_yaml(self, config_path: Path) -> None:
        """Save config to YAML file."""
        config_dict = self.__dict__
        save_yaml(config_dict, config_path)


@dataclass
class PredictionConfig:
    """Prediction configuration."""

    model: str = ""
    source: str = ""
    task: str = "detect"
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    device: str = "auto"
    save: bool = False
    project: str = "runs/predict"
    name: str = "exp"


@dataclass
class DataConfig:
    """Data preparation configuration."""

    source_dir: str = ""
    output_dir: str = ""
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    test_ratio: float = 0.1
    seed: int = 42


@dataclass
class CropConfig:
    """Crop extraction configuration."""

    dataset_path: str = ""
    output_dir: str = ""
    crop_size: Optional[tuple] = None
    padding: float = 0.1
    min_crop_size: int = 32
    preserve_splits: bool = True
    quality: int = 95
