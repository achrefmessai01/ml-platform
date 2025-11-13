import shutil
import os
import roboflow
from pathlib import Path
from PIL import Image
from glob import glob
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology


def setup_model(ontology: CaptionOntology, confidence: float = 0.2) -> GroundedSAM:
    """Set up the GroundedSAM model with the specified ontology and parameters."""
    model = GroundedSAM(ontology=ontology, box_threshold=confidence)
    return model


def preprocess_images(input_folder: str, max_resolution: int) -> None:
    """Convert all images to jpg and downsize if necessary."""
    image_files = glob(os.path.join(input_folder, "*"))

    for image_path in image_files:
        # Convert to JPG
        convert_to_jpg(image_path)

        # Downsize if necessary
        with Image.open(image_path) as img:
            width, height = img.size
            max_dim = max(width, height)
            if max_dim > max_resolution:
                scale = max_resolution / max_dim
                new_size = (int(width * scale), int(height * scale))
                resized_img = img.resize(new_size)
                resized_img.save(image_path)
                print(f"Resized image saved to {image_path} with size {new_size}")


def convert_to_jpg(image_path):
    """Convert an image to JPG format if it is not already in that format."""
    with Image.open(image_path) as img:
        # Convert to RGB (necessary for JPEG)
        rgb_img = img.convert("RGB")

        # Replace extension with .jpg
        jpg_path = os.path.splitext(image_path)[0] + ".jpg"

        # Save as JPEG (overwrite if same name)
        rgb_img.save(jpg_path, "JPEG", quality=95)

    # Remove the old file if it’s not already .jpg
    if jpg_path != image_path:
        os.remove(image_path)


def label_images(input_folder, model):
    """Auto label all images in the input folder and save to the output folder."""
    batches = glob(input_folder + "/*")

    for batch in batches:
        print(f"Processing batch: {batch}")
        labels = model.label(
            input_folder=batch, extension=".jpg", output_folder=input_folder
        )


def batch_images(input_folder: str, batch_size: int) -> None:
    """
    Copy all images from input folder into batched subdirectories.

    Args:
        input_folder (str): Path to the folder containing images
        batch_size (int): Number of images per batch subdirectory
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist")

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")

    image_files = [f for f in input_path.iterdir() if f.is_file()]

    # Sort files for consistent batching
    image_files.sort(key=lambda x: x.name)

    print(f"Found {len(image_files)} images. Creating batches of {batch_size}...")

    # Create batches and move files
    for i in range(0, len(image_files), batch_size):
        batch_num = (i // batch_size) + 10
        batch_folder = input_path / f"batch_{batch_num:03d}"
        batch_folder.mkdir(exist_ok=True)

        batch_files = image_files[i : i + batch_size]

        for image_file in batch_files:
            destination = batch_folder / image_file.name
            shutil.move(str(image_file), str(destination))

        print(f"Copied {len(batch_files)} images to {batch_folder.name}")

    print(
        f"Batching complete! Created {(len(image_files) - 1) // batch_size + 1} batches."
    )


def cleanup_batches(input_folder: str) -> None:
    """
    Remove all batch subdirectories from the input folder.

    Args:
        input_folder (str): Path to the folder containing batch subdirectories
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist")

    batch_folders = [
        f for f in input_path.iterdir() if f.is_dir() and f.name.startswith("batch_")
    ]

    print(f"Found {len(batch_folders)} batch folders. Removing...")

    for batch_folder in batch_folders:
        shutil.rmtree(batch_folder)
        print(f"Removed {batch_folder.name}")

    print("Cleanup complete!")


def upload_to_roboflow(dataset_folder: str, project_name: str) -> None:
    """Upload labeled images to Roboflow dataset."""
    rf = roboflow.Roboflow(api_key="MlW0ZIwhEkHcCdP281hK")
    workspace = rf.workspace("YOUR_WORKSPACE_URL")
    workspace.upload_dataset(
        dataset_folder,  # This is your dataset path
        num_workers=10,
        project_license="MIT",
        project_type="semantic-segmentation",
        batch_name=None,
        num_retries=0,
        is_prediction=False,  # optional, set to True if the dataset is not ground truth and needs approval
    )


if __name__ == "__main__":
    input_folder = "datasets/merged_data/all_images_labeled"
    batch_size = 300
    confidence = 0.2
    max_resolution = 1920
    ontology = CaptionOntology(
        {
            "leaf": "leaves",
        }
    )
    model = setup_model(ontology, confidence)
    preprocess_images(input_folder, max_resolution)
    batch_images(input_folder, batch_size)
    label_images(input_folder, model)
    cleanup_batches(input_folder)
