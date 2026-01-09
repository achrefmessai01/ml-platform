import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.optim import lr_scheduler


class LeafModel(pl.LightningModule):
    """A PyTorch Lightning module for leaf segmentation.

    This module uses a segmentation model from segmentation_models_pytorch to perform
    binary leaf segmentation on images. It handles training, validation, and testing
    with Dice loss and computes metrics like IoU, accuracy, and F1 score.

    Attributes
    ----------
    model : smp.Model
        The segmentation model architecture.
    std : torch.Tensor
        Standard deviation for image normalization.
    mean : torch.Tensor
        Mean values for image normalization.
    loss_fn : smp.losses.DiceLoss
        The loss function used for training.
    t_max : int
        Maximum number of iterations for the learning rate scheduler.
    """

    def __init__(self, arch: str, encoder_name: str, in_channels: int, out_classes: int, t_max: int, **kwargs) -> None:
        """Initialize the LeafModel with the specified architecture and parameters.

        Parameters
        ----------
        arch : str
            The architecture name for the segmentation model.
        encoder_name : str
            The encoder name for the model.
        in_channels : int
            Number of input channels.
        out_classes : int
            Number of output classes.
        t_max : int
            Maximum number of iterations for the learning rate scheduler.
        **kwargs
            Additional keyword arguments passed to the model creation.
        """
        super().__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.t_max = t_max

        # Logging hyperparameters
        self.save_hyperparameters()

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass for the leaf segmentation model.

        Normalizes the input image and passes it through the segmentation model.

        Parameters
        ----------
        image : torch.Tensor
            Input image tensor of shape (batch_size, num_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output mask tensor containing logits for each pixel.
        """
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch: tuple, stage: str) -> dict:
        """Compute loss and metrics for a batch of data.

        Parameters
        ----------
        batch : tuple
            A tuple containing the image and mask tensors.
        stage : str
            The stage name ('train', 'valid', or 'test') for logging purposes.

        Returns
        -------
        dict
            A dictionary containing the loss and metric values (tp, fp, fn, tn).
        """
        image = batch[0]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs: list, stage: str) -> None:
        """Compute and log metrics at the end of an epoch.

        Aggregates the metrics from all steps in an epoch and computes
        IoU, accuracy, and F1 score for logging.

        Parameters
        ----------
        outputs : list
            List of dictionaries containing metrics from each step.
        stage : str
            The stage name ('train', 'valid', or 'test') for logging purposes.
        """
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        # Compute accuracy and F1 score as well
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="micro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_accuracy": accuracy,
            f"{stage}_f1_score": f1_score,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch: tuple, batch_idx: int) -> dict:
        """Perform a training step for the leaf segmentation model.

        Computes the loss and metrics for a batch of training data and logs the training loss.

        Parameters
        ----------
        batch : tuple
            A tuple containing the image and mask tensors for the batch.
        batch_idx : int
            The index of the batch.

        Returns
        -------
        dict
            A dictionary containing the loss and metric values (tp, fp, fn, tn).
        """
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        self.log("train_loss", train_loss_info["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return train_loss_info

    def on_train_epoch_end(self) -> None:
        """Compute and log training metrics at the end of an epoch."""
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch: tuple, batch_idx: int) -> dict:
        """Perform a validation step for the leaf segmentation model."""
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        self.log("val_loss", valid_loss_info["loss"], prog_bar=True, on_step=False, on_epoch=True)
        return valid_loss_info

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at the end of an epoch."""
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch: tuple, batch_idx: int) -> dict:
        """Perform a test step for the leaf segmentation model."""
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at the end of an epoch."""
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self) -> dict:
        """Configure the optimizer and learning rate scheduler."""
        T_MAX = self.t_max
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
