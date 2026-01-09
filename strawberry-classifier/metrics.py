import torch
from torchmetrics import ConfusionMatrix


def create_confusion_matrix_from_predictions(
    preds: torch.Tensor, targets: torch.Tensor, num_classes: int, class_names: list[str] | None = None
) -> tuple[str, torch.Tensor]:
    """Create normalized confusion matrix from pre-collected predictions.

    This function generates a confusion matrix plot from already computed predictions
    and targets, avoiding the need for additional forward passes through the model.
    The matrix is normalized over true labels (rows sum to 1) to show recall values.

    Args:
        preds (torch.Tensor): Predicted class indices of shape (N,).
        targets (torch.Tensor): True class indices of shape (N,).
        num_classes (int): Total number of classes.
        class_names (list, optional): List of class names for labeling.
                                    If None, uses numeric labels.

    Returns:
        tuple: A tuple containing:
            - str: Path to the saved confusion matrix plot image.
            - torch.Tensor: The computed confusion matrix tensor.
    """
    # Initialize confusion matrix with normalization
    confmat = ConfusionMatrix(
        task="multiclass",
        num_classes=num_classes,
        normalize="true",  # Normalize over true labels (rows sum to 1)
    )

    # Compute confusion matrix directly from predictions
    confmat.update(preds, targets)
    cm = confmat.compute()

    # Use TorchMetrics native plotting
    fig, _ = confmat.plot(val=cm, labels=class_names)

    # Save plot
    cm_path = "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    fig.clear()

    return cm_path, cm
