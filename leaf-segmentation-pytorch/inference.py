import matplotlib.pyplot as plt
import torch
from torchvision.io import read_image
from torchvision.transforms import v2

from model import LeafModel


def main() -> None:
    """Run inference on a leaf image using the trained segmentation model.

    Loads a pre-trained LeafModel checkpoint, processes an input image,
    runs inference to generate a segmentation mask, and saves the results
    as a visualization.
    """
    # Configuration
    MODEL_CHECKPOINT_PATH = "best-model.ckpt"  # Update this path
    IMAGE_PATH = "dataset1_5d233ff4-IMG_9336_jpg.rf.62daba4b2dce52c85189137c341b73d5.jpg"  # Update this path
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    print("Loading model...")
    model = LeafModel.load_from_checkpoint(
        MODEL_CHECKPOINT_PATH,
        arch="FPN",
        encoder_name="resnet34",
        in_channels=3,
        out_classes=1,
        t_max=1,  # Dummy value, not used in inference
    )
    model.to(DEVICE)
    model.eval()

    # Load and preprocess the image
    print("Loading image...")
    image = read_image(IMAGE_PATH).float()  # Keep as uint8 equivalent, no /255.0 to match training

    # Apply transforms
    transforms = v2.Compose(
        [
            v2.Resize((256, 256), interpolation=v2.InterpolationMode.NEAREST),
        ]
    )
    image = transforms(image)

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(DEVICE)  # Add batch dimension
        logits = model(image_batch)
        pred_mask = logits.sigmoid().squeeze().cpu()

    print(f"Prediction mask range: {pred_mask.min().item():.4f} - {pred_mask.max().item():.4f}")

    # Visualize results
    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image.permute(1, 2, 0) / 255.0)  # Normalize for display
    plt.title("Input Image")
    plt.axis("off")

    # Predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(pred_mask.numpy(), cmap="gray")
    plt.title("Predicted Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig("inference_result.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("Inference completed. Result saved as 'inference_result.png'.")


if __name__ == "__main__":
    main()
