import shutil
from pathlib import Path


def batch_images(input_folder: str, batch_size: int) -> None:
    """
    Move all images from input folder into batched subdirectories.

    Args:
        input_folder (str): Path to the folder containing images
        batch_size (int): Number of images per batch subdirectory
    """
    input_path = Path(input_folder)

    if not input_path.exists():
        raise FileNotFoundError(f"Input folder '{input_folder}' does not exist")

    if batch_size <= 0:
        raise ValueError("Batch size must be greater than 0")

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
    image_files = [f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"No image files found in '{input_folder}'")
        return

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

        print(f"Moved {len(batch_files)} images to {batch_folder.name}")

    print(f"Batching complete! Created {(len(image_files) - 1) // batch_size + 1} batches.")


if __name__ == "__main__":
    # Example usage
    folder_path = input("Enter the path to the image folder: ")
    batch_size = int(input("Enter the batch size: "))

    try:
        batch_images(folder_path, batch_size)
    except Exception as e:
        print(f"Error: {e}")
