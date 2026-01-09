import random
import shutil
from pathlib import Path
from typing import Dict, Union

from tqdm import tqdm

from .utils import load_yaml, save_yaml


def merge_datasets(
    dataset1_path: Union[str, Path],
    dataset2_path: Union[str, Path],
    output_path: Union[str, Path],
    dataset1_name: str = "dataset1",
    dataset2_name: str = "dataset2",
) -> Dict:
    """Merge two YOLO datasets."""
    dataset1_path = Path(dataset1_path)
    dataset2_path = Path(dataset2_path)
    output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    data1 = load_yaml(str(dataset1_path / "data.yaml")) if (dataset1_path / "data.yaml").exists() else {}
    data2 = load_yaml(str(dataset2_path / "data.yaml")) if (dataset2_path / "data.yaml").exists() else {}

    class_names1 = data1.get("names", [])
    class_names2 = data2.get("names", [])

    all_classes = []
    class_mapping2 = {}

    for cls in class_names1:
        if cls not in all_classes:
            all_classes.append(cls)

    for i, cls in enumerate(class_names2):
        if cls in all_classes:
            class_mapping2[i] = all_classes.index(cls)
        else:
            all_classes.append(cls)
            class_mapping2[i] = len(all_classes) - 1

    stats = {"images": 0, "labels": 0}

    for split in ["train", "val", "test"]:
        output_images_dir = output_path / split / "images"
        output_labels_dir = output_path / split / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        for dataset_path, dataset_name, class_mapping in [
            (dataset1_path, dataset1_name, {}),
            (dataset2_path, dataset2_name, class_mapping2),
        ]:
            images_dir = dataset_path / split / "images"
            labels_dir = dataset_path / split / "labels"

            if not images_dir.exists():
                continue

            for image_file in tqdm(
                list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")),
                desc=f"Merging {dataset_name} {split}",
            ):
                new_image_name = f"{dataset_name}_{image_file.name}"
                shutil.copy2(image_file, output_images_dir / new_image_name)
                stats["images"] += 1

                label_file = labels_dir / f"{image_file.stem}.txt"
                if label_file.exists():
                    new_labels = []

                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                old_class_id = int(parts[0])
                                new_class_id = class_mapping.get(old_class_id, old_class_id)
                                new_labels.append(f"{new_class_id} {' '.join(parts[1:])}")

                    new_label_file = output_labels_dir / f"{Path(new_image_name).stem}.txt"
                    with open(new_label_file, "w") as f:
                        f.write("\n".join(new_labels) + "\n" if new_labels else "")

                    stats["labels"] += len(new_labels)

    merged_data = {
        "path": str(output_path),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(all_classes),
        "names": all_classes,
    }

    save_yaml(merged_data, str(output_path / "data.yaml"))

    with open(output_path / "classes.txt", "w") as f:
        f.write("\n".join(all_classes) + "\n")

    return {
        "images": stats["images"],
        "labels": stats["labels"],
        "classes": len(all_classes),
    }


def create_subset(
    dataset_path: Union[str, Path],
    output_path: Union[str, Path],
    subset_ratio: float = 0.1,
    random_seed: int = 42,
) -> Dict:
    """Create a subset of a YOLO dataset."""
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    random.seed(random_seed)

    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
        data = load_yaml(str(data_yaml))
        data.get("names", [])
    else:
        pass

    stats = {"total": 0, "selected": 0}

    for split in ["train", "val", "test"]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        if not images_dir.exists():
            continue

        output_images_dir = output_path / split / "images"
        output_labels_dir = output_path / split / "labels"
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

        image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
        stats["total"] += len(image_files)

        subset_size = max(1, int(len(image_files) * subset_ratio))
        selected_images = random.sample(image_files, subset_size)
        stats["selected"] += len(selected_images)

        for image_file in tqdm(selected_images, desc=f"Creating {split} subset"):
            shutil.copy2(image_file, output_images_dir / image_file.name)

            label_file = labels_dir / f"{image_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, output_labels_dir / label_file.name)

    if data_yaml.exists():
        subset_data = load_yaml(str(data_yaml))
        subset_data["path"] = str(output_path)
        save_yaml(subset_data, str(output_path / "data.yaml"))

    classes_txt = dataset_path / "classes.txt"
    if classes_txt.exists():
        shutil.copy2(classes_txt, output_path / "classes.txt")

    return stats
