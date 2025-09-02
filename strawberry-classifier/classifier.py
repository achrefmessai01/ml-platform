import os
import shutil
import tempfile

import mlflow
import mlflow.pytorch
import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

from metrics import create_confusion_matrix_from_predictions


class StrawberryClassifier(pl.LightningModule):
    """PyTorch Lightning module for strawberry classification using EfficientNet-V2-M.

    This module implements a 6-class strawberry classifier with automatic device management,
    progress tracking, and validation prediction collection for confusion matrix generation.

    Args:
        config (dict): Configuration dictionary containing model and training parameters.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # EfficientNet backbone
        self.model = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.DEFAULT
        )
        # EfficientNet-V2-M has 1280 features before the classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=config["dropout_rate"], inplace=True),
            nn.Linear(1280, config["num_classes"]),
        )

        self.criterion = nn.CrossEntropyLoss()

        # For collecting validation predictions efficiently
        self.validation_step_outputs: list = []

    def forward(self, x):
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Logits for each class of shape (batch_size, num_classes).
        """
        return self.model(x)

    def training_step(self, batch, _):
        """Training step for one batch.

        Args:
            batch: Tuple of (images, targets) from the training dataloader.
            _: Batch index (unused).

        Returns:
            torch.Tensor: Training loss for this batch.
        """
        imgs, targets = batch
        outputs = self.forward(imgs)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        """Validation step for one batch.

        Computes validation loss and accuracy, and collects predictions
        for confusion matrix generation.

        Args:
            batch: Tuple of (images, targets) from the validation dataloader.
            _: Batch index (unused).

        Returns:
            torch.Tensor: Validation loss for this batch.
        """
        imgs, targets = batch
        outputs = self.forward(imgs)
        loss = self.criterion(outputs, targets)
        acc = (outputs.argmax(dim=1) == targets).float().mean()
        preds = outputs.argmax(dim=1)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        # Collect predictions and targets for confusion matrix
        self.validation_step_outputs.append(
            {"preds": preds.detach().cpu(), "targets": targets.detach().cpu()}
        )

        return loss

    def on_validation_epoch_start(self):
        """Called at the start of each validation epoch.

        Clears previous epoch's predictions to ensure only the final
        validation epoch's predictions are used for confusion matrix.
        """
        # Clear previous epoch's predictions at the start of each validation epoch
        self.validation_step_outputs.clear()

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch.

        Keeps the predictions from the final validation epoch for confusion matrix
        generation. No cleanup is performed here to preserve the final predictions.
        """
        # Keep the predictions from the final validation epoch for confusion matrix
        pass

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler for training.

        Returns:
            dict: Dictionary containing optimizer and lr_scheduler configuration.
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.config["learning_rate"]
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config["max_epochs"],
            eta_min=self.config["min_learning_rate"],
        )
        return [optimizer], [scheduler]


def train_classifier(config: dict):
    """Train a strawberry classifier with MLflow tracking and model registry.

    This function handles the complete training pipeline including:
    - Data loading and preprocessing
    - Model initialization and training with PyTorch Lightning
    - Early stopping and best model checkpointing
    - Confusion matrix generation and logging
    - Model registration in MLflow Model Registry
    - Temporary file cleanup

    Args:
        config (dict): Configuration dictionary containing all training parameters:
            - train_dir, val_dir: Dataset paths
            - num_classes, learning_rate, dropout_rate: Model parameters
            - batch_size, max_epochs, num_workers: Training parameters
            - early_stopping_*: Early stopping configuration
            - mlflow_uri, experiment_name: MLflow settings
            - model_registry_name: Name for model registry

    Returns:
        StrawberryClassifier: The best trained model loaded from checkpoint.
    """
    # Data setup
    weights = models.EfficientNet_V2_M_Weights.DEFAULT
    base_transform = weights.transforms()

    # Training transforms with RandAugment
    train_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_size"], config["image_size"])
            ),  # Ensure consistent size
            transforms.RandAugment(
                num_ops=config[
                    "randaugment_num_ops"
                ],  # Number of augmentation operations to apply
                magnitude=config[
                    "randaugment_magnitude"
                ],  # Magnitude of augmentation (0-10 scale)
                num_magnitude_bins=config[
                    "randaugment_num_magnitude_bins"
                ],  # Number of magnitude bins
            ),
            base_transform,  # Apply normalization after augmentation
        ]
    )

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_size"], config["image_size"])
            ),  # Ensure consistent size for validation
            base_transform,
        ]
    )

    train_dataset = ImageFolder(config["train_dir"], transform=train_transform)
    val_dataset = ImageFolder(config["val_dir"], transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], num_workers=config["num_workers"]
    )

    # Model
    model = StrawberryClassifier(config=config)

    # Set up MLflow tracking URI and experiment - this handles existing experiments automatically
    mlflow.set_tracking_uri(config["mlflow_uri"])
    experiment = mlflow.set_experiment(config["experiment_name"])

    # MLflow logger using experiment_name (PyTorch Lightning requirement)
    mlflow_kwargs = {
        "experiment_name": config["experiment_name"],
        "tracking_uri": config["mlflow_uri"],
    }

    # Add run_name if specified in config
    if "run_name" in config and config["run_name"]:
        mlflow_kwargs["run_name"] = config["run_name"]

    mlf_logger = MLFlowLogger(**mlflow_kwargs)  # Early stopping callback
    early_stop_callback = EarlyStopping(
        monitor=config["early_stopping_monitor"],
        min_delta=config["early_stopping_min_delta"],
        patience=config["early_stopping_patience"],
        verbose=False,
        mode=config["early_stopping_mode"],
    )

    # Create temporary directory for checkpoints
    temp_dir = tempfile.mkdtemp(prefix="lightning_checkpoints_")

    # Model checkpoint callback to save best model
    checkpoint_callback = ModelCheckpoint(
        monitor=config["early_stopping_monitor"],
        mode=config["early_stopping_mode"],
        save_top_k=1,
        dirpath=temp_dir,
        filename="best_model",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        logger=mlf_logger,
        accelerator="auto",
        devices="auto",
        callbacks=[early_stop_callback, checkpoint_callback],
        accumulate_grad_batches=config[
            "accumulate_grad_batches"
        ],  # Gradient accumulation
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Load the best model for MLflow saving
    best_model = StrawberryClassifier.load_from_checkpoint(
        checkpoint_callback.best_model_path, config=config
    )

    # Create confusion matrix from collected validation predictions (use original model)
    class_names = train_dataset.classes if hasattr(train_dataset, "classes") else None

    if model.validation_step_outputs:
        # Concatenate all collected predictions and targets
        all_preds = torch.cat(
            [output["preds"] for output in model.validation_step_outputs]
        )
        all_targets = torch.cat(
            [output["targets"] for output in model.validation_step_outputs]
        )

        # Create confusion matrix
        cm_path, _ = create_confusion_matrix_from_predictions(
            all_preds, all_targets, config["num_classes"], class_names
        )

        # Log confusion matrix as artifact
        mlf_logger.experiment.log_artifact(mlf_logger.run_id, cm_path)

        # Clean up temporary file
        if os.path.exists(cm_path):
            os.remove(cm_path)
    else:
        print("No validation predictions collected for confusion matrix")

    # Register model in MLflow Model Registry
    model_name = config.get("model_registry_name", "strawberry-classifier")

    # Get a sample input for model signature inference (ensure correct dtype)
    sample_batch = next(iter(val_loader))
    sample_input = sample_batch[0][:1]  # Take first image from batch
    # Convert to float32 numpy for MLflow
    sample_input_numpy = sample_input.detach().cpu().float().numpy()

    # Put best model in eval mode and move to CPU for MLflow compatibility
    best_model.eval()
    best_model = best_model.cpu()

    best_loss = checkpoint_callback.best_model_score
    loss_value = best_loss.item() if hasattr(best_loss, "item") else float(best_loss)
    # Log the best model as an MLflow artifact and register it
    with mlflow.start_run(run_id=mlf_logger.run_id):
        mlflow.pytorch.log_model(
            pytorch_model=best_model,
            name="model",  # Updated from artifact_path to name
            registered_model_name=model_name,
            # Float32 numpy array for MLflow compatibility
            input_example=sample_input_numpy,
        )
        mlflow.log_metric("best_val_loss", loss_value)

    print(f"Model registered in MLflow Model Registry as: {model_name}")

    # Clean up temporary checkpoint directory
    shutil.rmtree(temp_dir)

    return best_model


if __name__ == "__main__":
    # Configuration dictionary - update these settings as needed
    config = {
        # Data paths
        "train_dir": "datasets/crop_cell/classification/farm_dataset_obb_6_classes/train",
        "val_dir": "datasets/crop_cell/classification/farm_dataset_obb_6_classes/val",
        # Data preprocessing hyperparameters
        "image_size": 224,
        "randaugment_num_ops": 2,
        "randaugment_magnitude": 9,
        "randaugment_num_magnitude_bins": 31,
        # Model hyperparameters
        "num_classes": 6,
        "learning_rate": 1e-5,
        "min_learning_rate": 1e-7,  # Minimum LR for cosine annealing
        "dropout_rate": 0.30,
        # Training hyperparameters
        "batch_size": 8,
        "max_epochs": 50,
        "num_workers": 4,
        "accumulate_grad_batches": 4,  # Gradient accumulation steps
        # Early stopping
        "early_stopping_monitor": "val_loss",
        "early_stopping_min_delta": 0.001,
        "early_stopping_patience": 5,
        "early_stopping_mode": "min",
        # MLflow settings
        "mlflow_uri": "http://localhost:5011",
        "experiment_name": "strawberry-classifier",
        "run_name": None,
        "model_registry_name": "strawberry-classifier",
    }

    trained_model = train_classifier(config)
