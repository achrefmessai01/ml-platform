import copy
import os
import shutil
import tempfile
from datetime import datetime

import mlflow
import mlflow.pytorch
import torch
from detectron2 import model_zoo
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data import get_detection_dataset_dicts
from detectron2.data import transforms as T
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.engine.hooks import BestCheckpointer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger

setup_logger()


class MLflowHook(HookBase):
    """
    Hook to log metrics and register models with MLflow.
    """

    def __init__(self, cfg, model, hyperparams):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.hyperparams = hyperparams
        self.run_id = None

    def before_train(self):
        # Start MLflow run
        mlflow.set_tracking_uri(self.hyperparams["mlflow_tracking_uri"])
        mlflow.set_experiment(self.hyperparams["mlflow_experiment"])

        run_name = f"faster-rcnn-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        run = mlflow.start_run(run_name=run_name)
        self.run_id = run.info.run_id

        # Log all hyperparameters to MLflow
        for param_name, param_value in self.hyperparams.items():
            # Convert lists/tuples to strings for MLflow compatibility
            if isinstance(param_value, (list, tuple)):
                param_value = str(param_value)
            mlflow.log_param(param_name, param_value)

        print(f"Logged {len(self.hyperparams)} hyperparameters to MLflow")

    def after_step(self):
        # Log metrics from storage
        storage = self.trainer.storage
        for key, (value, iteration) in storage.latest().items():
            mlflow.log_metric(key, value, step=iteration)

    def after_train(self):
        # Log config as artifact to MLflow
        config_path = os.path.join(self.cfg.OUTPUT_DIR, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(self.cfg.dump())
        mlflow.log_artifact(config_path)

        # Log the actual model weights file
        model_path = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
        if os.path.exists(model_path):
            try:
                # Log the .pth file as an artifact (simpler and more reliable for Detectron2)
                mlflow.log_artifact(model_path, "model_weights")
                print(f"Logged model weights: {model_path}")

                # Also try to register the model using pytorch.log_model for compatibility
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    artifact_path="model",
                    registered_model_name=self.hyperparams["registered_model_name"],
                )

                print(
                    f"Model registered in MLflow Model Registry as '{self.hyperparams['registered_model_name']}'"
                )
            except Exception as e:  # noqa: BLE001
                print(f"Warning: Could not register model: {e}")
        else:
            print(f"Warning: Model file not found at {model_path}")

        mlflow.end_run()

    def _get_best_validation_ap(self):
        """Get the best validation AP from storage."""
        storage = self.trainer.storage
        try:
            if "bbox/AP" in storage.latest():
                return storage.latest()["bbox/AP"][0]
        except (KeyError, AttributeError):
            pass
        return 0.0


def custom_mapper(dataset_dict, hyperparams):
    """
    Custom mapper for data augmentation using hyperparameters.

    Args:
        dataset_dict: Dataset dictionary in detectron2 format
        hyperparams: Dictionary containing augmentation hyperparameters
    """
    # Make a deep copy to avoid modifying the original dataset_dict
    dataset_dict = copy.deepcopy(dataset_dict)

    # Read image
    image = utils.read_image(dataset_dict["file_name"], format=hyperparams["image_format"])

    # Define augmentations using hyperparameters
    augs = T.AugmentationList(
        [
            T.RandomFlip(
                prob=hyperparams["aug_horizontal_flip_prob"],
                horizontal=True,
                vertical=False,
            ),
            T.RandomFlip(
                prob=hyperparams["aug_vertical_flip_prob"],
                horizontal=False,
                vertical=True,
            ),
            T.RandomRotation(angle=hyperparams["aug_rotation_angle_range"]),
            T.RandomBrightness(*hyperparams["aug_brightness_range"]),
            T.RandomContrast(*hyperparams["aug_contrast_range"]),
            T.RandomSaturation(*hyperparams["aug_saturation_range"]),
            T.RandomLighting(hyperparams["aug_lighting_scale"]),
            # Optionally add more augmentations:
            # T.RandomCrop("relative_range", (0.8, 1.0)),  # Random crops
        ]
    )

    # Apply augmentation using standard detectron2 pattern
    # This follows the exact pattern from the official documentation
    aug_input = T.AugInput(image)
    transforms = augs(aug_input)

    # Get the transformed image from the AugInput object
    image = aug_input.image

    # Transform annotations using the same transforms applied to the image
    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
    ]

    return {
        "image": torch.as_tensor(image.transpose(2, 0, 1).astype("float32")),
        "instances": utils.annotations_to_instances(annos, image.shape[:2]),
        "height": image.shape[0],
        "width": image.shape[1],
    }


def build_train_loader(cfg, hyperparams):
    """
    Build training data loader with custom augmentations.

    Args:
        cfg: Detectron2 config object
        hyperparams: Dictionary containing all hyperparameters including augmentation settings
    """

    # Create a mapper function that uses the hyperparameters
    def mapper_with_hyperparams(dataset_dict):
        return custom_mapper(dataset_dict, hyperparams)

    # Get dataset dicts for the training datasets
    dataset_dicts = get_detection_dataset_dicts(cfg.DATASETS.TRAIN)

    return build_detection_train_loader(
        dataset_dicts,
        mapper=mapper_with_hyperparams,
        total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
        aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
    )


class CustomTrainer(DefaultTrainer):
    """
    Custom trainer with COCO evaluation and hyperparameter-driven data augmentation.
    """

    # Class variable to store hyperparameters
    _hyperparams = None

    def __init__(self, cfg, hyperparams):
        """
        Initialize CustomTrainer with hyperparameters.

        Args:
            cfg: Detectron2 config object
            hyperparams: Dictionary containing all hyperparameters
        """
        self.hyperparams = hyperparams
        # Also set as class variable for class methods
        CustomTrainer._hyperparams = hyperparams
        super().__init__(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build training data loader using stored hyperparameters.

        Args:
            cfg: Detectron2 config object

        Returns:
            DataLoader with custom augmentations
        """
        if cls._hyperparams is None:
            raise ValueError(
                "Hyperparameters not set. Call CustomTrainer.set_hyperparams() first."
            )
        return build_train_loader(cfg, cls._hyperparams)

    @classmethod
    def set_hyperparams(cls, hyperparams):
        """
        Set hyperparameters for the trainer class.

        Args:
            hyperparams: Dictionary containing all hyperparameters including augmentation settings
        """
        cls._hyperparams = hyperparams

    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        """
        Build evaluator for validation.

        Args:
            cfg: Detectron2 config object
            dataset_name: Name of the validation dataset

        Returns:
            COCOEvaluator instance
        """
        return COCOEvaluator(dataset_name, output_dir=cfg.OUTPUT_DIR)


def main():
    # Hyperparameters dictionary - all training parameters in one place
    hyperparams = {
        # Model architecture
        "model_config": "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml",
        "model_architecture": "faster_rcnn_R_50_FPN_3x",
        "backbone": "ResNet-50-FPN",
        "framework": "Detectron2",
        # Dataset parameters
        "train_dataset": "strawberries_train",
        "val_dataset": "strawberries_val",
        "train_annotations": "datasets/merged_data/detection/combined_coco/annotations/instances_train.json",
        "train_images": "datasets/merged_data/detection/combined_coco/train",
        "val_annotations": "datasets/merged_data/detection/combined_coco/annotations/instances_val.json",
        "val_images": "datasets/merged_data/detection/combined_coco/val",
        # Model parameters
        "num_classes": 1,  # Only strawberry class
        "roi_batch_size_per_image": 256,  # Reduced from 512 to save VRAM
        # Training parameters
        "batch_size": 2,  # Reduced from 4 to save VRAM
        "learning_rate": 0.00125,  # Adjusted for smaller batch size (0.0025 * 2/4)
        "gradient_accumulation_steps": 2,  # Accumulate gradients to simulate larger batch
        "max_iterations": 2500,  # Reasonable for initial training
        "lr_decay_steps": [1500, 2000],  # Decay at 67% and 90%
        "warmup_iterations": 300,  # Warmup for first ~17% of training
        "checkpoint_period": 500,
        # Evaluation parameters
        "eval_period": 400,  # Evaluate less frequently to save VRAM
        "num_workers": 1,  # Reduced workers to save memory
        # Data augmentation parameters
        "image_format": "RGB",  # Image format: "RGB" or "BGR"
        "input_min_size_train": (480, 600),  # Reduced image sizes to save VRAM
        "input_max_size_train": 800,  # Reduced from 1200 to save VRAM
        "aug_horizontal_flip_prob": 0.5,  # Probability of horizontal flip
        "aug_vertical_flip_prob": 0.1,  # Probability of vertical flip (rare but useful)
        "aug_rotation_angle_range": [-15, 15],  # Rotation angle range in degrees
        "aug_brightness_range": [0.8, 1.2],  # Brightness adjustment range
        "aug_contrast_range": [0.8, 1.2],  # Contrast adjustment range
        "aug_saturation_range": [0.8, 1.2],  # Saturation adjustment range
        "aug_lighting_scale": 0.7,  # PCA-based color jitter scale
        # MLflow parameters
        "mlflow_tracking_uri": "http://localhost:5011",
        "mlflow_experiment": "strawberry-detection",
        "registered_model_name": "strawberry-detector-faster-rcnn",
    }

    # Register datasets
    register_coco_instances(
        hyperparams["train_dataset"],
        {},
        hyperparams["train_annotations"],
        hyperparams["train_images"],
    )
    register_coco_instances(
        hyperparams["val_dataset"],
        {},
        hyperparams["val_annotations"],
        hyperparams["val_images"],
    )

    # Configuration
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(hyperparams["model_config"]))

    # Datasets
    cfg.DATASETS.TRAIN = (hyperparams["train_dataset"],)
    cfg.DATASETS.TEST = (hyperparams["val_dataset"],)

    # Model
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(hyperparams["model_config"])
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = hyperparams["num_classes"]
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = hyperparams["roi_batch_size_per_image"]
    
    cfg.MODEL.BACKBONE.FREEZE_AT = 2
    if hasattr(cfg.MODEL, 'RESNETS') and hasattr(cfg.MODEL.RESNETS, 'CHECKPOINT'):
        cfg.MODEL.RESNETS.CHECKPOINT = True

    # Solver
    cfg.SOLVER.IMS_PER_BATCH = hyperparams["batch_size"]
    cfg.SOLVER.BASE_LR = hyperparams["learning_rate"]
    cfg.SOLVER.MAX_ITER = hyperparams["max_iterations"]
    cfg.SOLVER.STEPS = hyperparams["lr_decay_steps"]
    cfg.SOLVER.WARMUP_ITERS = hyperparams["warmup_iterations"]
    cfg.SOLVER.CHECKPOINT_PERIOD = hyperparams["checkpoint_period"]
    cfg.SOLVER.LR_SCHEDULER_NAME = "WarmupCosineLR"
    cfg.SOLVER.AMP.ENABLED = True

    # Evaluation
    cfg.TEST.EVAL_PERIOD = hyperparams["eval_period"]
    cfg.DATALOADER.NUM_WORKERS = hyperparams["num_workers"]

    # Input
    cfg.INPUT.FORMAT = hyperparams["image_format"]  # Set image format (RGB or BGR)
    cfg.INPUT.MIN_SIZE_TRAIN = hyperparams["input_min_size_train"]  # Configurable image sizes
    cfg.INPUT.MAX_SIZE_TRAIN = hyperparams["input_max_size_train"]

    # Use temporary directory for output (will be cleaned up after MLflow logging)
    temp_dir = tempfile.mkdtemp(prefix="detectron2_training_")
    cfg.OUTPUT_DIR = temp_dir
    print(f"Using temporary output directory: {temp_dir}")

    # Set hyperparameters for data augmentation in the CustomTrainer
    CustomTrainer.set_hyperparams(hyperparams)

    # Create trainer
    trainer = CustomTrainer(cfg, hyperparams)

    # Build model to access it for hooks and checkpointer
    model = trainer.build_model(cfg)

    # Add MLflow hook with hyperparameters
    mlflow_hook = MLflowHook(cfg, model, hyperparams)
    trainer.register_hooks([mlflow_hook])

    # Add BestCheckpointer for early stopping based on bbox/AP
    checkpointer = BestCheckpointer(
        checkpointer=DetectionCheckpointer(model, cfg.OUTPUT_DIR),
        val_metric="bbox/AP",
        mode="max",
        eval_period=cfg.TEST.EVAL_PERIOD,
    )
    trainer.register_hooks([checkpointer])

    # Load pretrained weights
    trainer.resume_or_load(resume=False)

    # Train
    trainer.train()

    # Clean up temporary directory after logging to MLflow
    if os.path.exists(cfg.OUTPUT_DIR):
        try:
            shutil.rmtree(cfg.OUTPUT_DIR)
        except Exception as e:  # noqa: BLE001
            print(f"Warning: Could not clean up temporary directory: {e}")


if __name__ == "__main__":
    main()
