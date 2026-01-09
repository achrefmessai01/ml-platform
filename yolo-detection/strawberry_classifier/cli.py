import argparse
import sys

from .core.classifier import StrawberryClassifier
from .core.detector import StrawberryDetector
from .data.analyzer import analyze_dataset, compare_datasets
from .data.color_transfer import apply_color_transfer_to_dataset
from .data.crop_extractor import CropExtractor
from .data.dataset_ops import create_subset, merge_datasets
from .data.prepare import prepare_dataset
from .data.validator import validate_dataset
from .visualization.plot_utils import visualize_dataset_obb_samples


def prepare_data_command(args: argparse.Namespace) -> None:
    """Prepare dataset command."""
    prepare_dataset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        split_ratios=(args.train_ratio, args.val_ratio, args.test_ratio),
        seed=args.seed,
    )
    print(f"Dataset prepared successfully: {args.output_dir}")


def extract_crops_command(args: argparse.Namespace) -> None:
    """Extract crops command."""
    extractor = CropExtractor(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        crop_size=(args.crop_width, args.crop_height) if args.crop_width and args.crop_height else None,
        padding=args.padding,
        min_crop_size=args.min_crop_size,
        preserve_splits=args.preserve_splits,
        quality=args.quality,
    )
    extractor.extract_crops()
    stats = extractor.get_stats()
    print(f"Extracted {stats['total_crops']} crops")
    for class_name, count in stats["crops_per_class"].items():
        print(f"  {class_name}: {count}")


def color_transfer_command(args: argparse.Namespace) -> None:
    """Color transfer command."""
    stats = apply_color_transfer_to_dataset(
        dataset_path=args.dataset,
        source_reference=args.reference,
        output_path=args.output,
        probability=args.probability,
        intensity=args.intensity,
    )
    print("Color transfer complete:")
    print(f"  Processed: {stats['processed']} images")
    print(f"  Skipped: {stats['skipped']} images")


def compare_datasets_command(args: argparse.Namespace) -> None:
    """Compare datasets command."""
    results = compare_datasets(args.dataset1, args.dataset2)

    print(f"\nDataset Comparison: {args.name1} vs {args.name2}")
    print("=" * 60)

    for split1, split_results in results.items():
        print(f"\n{split1.upper()} split from {args.name1}:")
        print("-" * 30)

        for split2, stats in split_results.items():
            print(
                f"  vs {split2}: {stats['overlap_count']}/{stats['dataset1_count']} "
                f"({stats['overlap_percentage']:.1f}%) images overlap"
            )


def analyze_dataset_command(args: argparse.Namespace) -> None:
    """Analyze dataset command."""
    analysis = analyze_dataset(args.dataset)

    print(f"\nDataset Analysis: {args.dataset}")
    print("=" * 50)

    for split_name, counts in analysis["split_counts"].items():
        if not counts:
            continue

        print(f"\n{split_name.upper()} Split:")
        print("-" * 20)

        total_labels = sum(counts.values())
        print(f"Total labels: {total_labels}")

        for class_id in sorted(counts.keys()):
            class_name = (
                analysis["class_names"][class_id] if class_id < len(analysis["class_names"]) else f"Class_{class_id}"
            )
            count = counts[class_id]
            percentage = (count / total_labels) * 100
            print(f"  {class_name}: {count} ({percentage:.1f}%)")


def validate_dataset_command(args: argparse.Namespace) -> None:
    """Validate dataset command."""
    result = validate_dataset(args.dataset)

    print(f"Validating dataset: {args.dataset}")
    print("=" * 50)

    if result["structure_issues"]:
        print("\nStructure Issues:")
        for issue in result["structure_issues"]:
            print(f"  ❌ {issue}")
        return
    else:
        print("✅ Dataset structure is valid")

    if result["images_issues"]:
        print("\nImage/Label Issues:")
        for issue in result["images_issues"]:
            print(f"  ⚠️  {issue}")
    else:
        print("✅ All images have corresponding labels")

    if result["label_issues"]:
        print("\nLabel Format Issues:")
        for issue in result["label_issues"][:10]:
            print(f"  ❌ {issue}")
        if len(result["label_issues"]) > 10:
            print(f"  ... and {len(result['label_issues']) - 10} more issues")
    else:
        print("✅ All labels have valid format")

    print("\nDataset Statistics:")
    stats = result["stats"]
    print(f"  Images: {stats['images']}")
    print(f"  Labels: {stats['labels']}")
    print(f"  Orphaned images: {stats['orphaned_images']}")
    print(f"  Orphaned labels: {stats['orphaned_labels']}")

    if result["class_counts"]:
        print("\nClass Distribution:")
        for class_id in sorted(result["class_counts"].keys()):
            print(f"  Class {class_id}: {result['class_counts'][class_id]} instances")

    if result["valid"]:
        print("\n🎉 Dataset validation passed!")
    else:
        print(f"\n⚠️  Found {result['total_issues']} issues")


def merge_datasets_command(args: argparse.Namespace) -> None:
    """Merge datasets command."""
    stats = merge_datasets(
        dataset1_path=args.dataset1,
        dataset2_path=args.dataset2,
        output_path=args.output,
        dataset1_name=args.name1,
        dataset2_name=args.name2,
    )
    print("Merge complete:")
    print(f"  Total images: {stats['images']}")
    print(f"  Total labels: {stats['labels']}")
    print(f"  Classes: {stats['classes']}")


def visualize_obb_command(args: argparse.Namespace) -> None:
    """Visualize oriented bounding boxes command."""
    visualize_dataset_obb_samples(
        dataset_path=args.dataset,
        num_samples=args.num_samples,
        split=args.split,
        save_path=args.output if args.save else None,
        figsize=(args.width, args.height),
    )
    if args.save:
        print(f"OBB visualization saved to: {args.output}")
    else:
        print("OBB visualization displayed")


def create_subset_command(args: argparse.Namespace) -> None:
    """Create subset command."""
    stats = create_subset(
        dataset_path=args.dataset,
        output_path=args.output,
        subset_ratio=args.ratio,
        random_seed=args.seed,
    )
    print("Subset created:")
    print(f"  Original: {stats['total']} images")
    print(f"  Subset: {stats['selected']} images ({args.ratio * 100:.1f}%)")


def train_command(args: argparse.Namespace) -> None:
    """Train model command."""
    if args.task == "detect":
        model = StrawberryDetector(args.model, device=args.device)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            patience=args.patience,
        )
    else:  # classify
        model = StrawberryClassifier(args.model, device=args.device)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.imgsz,
            project=args.project,
            name=args.name,
            patience=args.patience,
        )

    print(f"Training completed: {results.save_dir}")


def predict_command(args: argparse.Namespace) -> None:
    """Predict command."""
    if args.task == "detect":
        model = StrawberryDetector(args.model, device=args.device)
    else:  # classify
        model = StrawberryClassifier(args.model, device=args.device)

    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=args.save,
        project=args.project,
        name=args.name,
    )

    if results:
        print(f"Prediction completed: {results[0].save_dir}")


def validate_model_command(args: argparse.Namespace) -> None:
    """Validate model command."""
    if args.task == "detect":
        model = StrawberryDetector(args.model, device=args.device)
    else:  # classify
        model = StrawberryClassifier(args.model, device=args.device)

    results = model.validate(
        data=args.data,
        batch=args.batch_size,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
    )

    print(f"Validation completed: {results.save_dir}")


def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description="Strawberry Classifier CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Data preparation commands
    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare dataset")
    prepare_parser.add_argument("--source-dir", required=True, help="Source dataset directory")
    prepare_parser.add_argument("--output-dir", required=True, help="Output directory")
    prepare_parser.add_argument("--train-ratio", type=float, default=0.7, help="Training ratio")
    prepare_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio")
    prepare_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test ratio")
    prepare_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    extract_parser = subparsers.add_parser("extract-crops", help="Extract crops for classification")
    extract_parser.add_argument("--dataset-path", required=True, help="Dataset path")
    extract_parser.add_argument("--output-dir", required=True, help="Output directory")
    extract_parser.add_argument("--crop-width", type=int, help="Crop width")
    extract_parser.add_argument("--crop-height", type=int, help="Crop height")
    extract_parser.add_argument("--padding", type=float, default=0.1, help="Padding around bbox")
    extract_parser.add_argument("--min-crop-size", type=int, default=32, help="Minimum crop size")
    extract_parser.add_argument("--preserve-splits", action="store_true", default=True, help="Preserve splits")
    extract_parser.add_argument("--quality", type=int, default=95, help="JPEG quality")

    # Dataset utility commands
    color_parser = subparsers.add_parser("color-transfer", help="Apply color transfer to dataset")
    color_parser.add_argument("--dataset", required=True, help="Path to input dataset")
    color_parser.add_argument("--reference", required=True, help="Path to reference image/directory")
    color_parser.add_argument("--output", required=True, help="Path to output dataset")
    color_parser.add_argument(
        "--probability",
        type=float,
        default=0.5,
        help="Probability of applying color transfer",
    )
    color_parser.add_argument("--intensity", type=float, default=1.0, help="Intensity of color transfer")

    compare_parser = subparsers.add_parser("compare-datasets", help="Compare two datasets for overlap")
    compare_parser.add_argument("--dataset1", required=True, help="Path to first dataset")
    compare_parser.add_argument("--dataset2", required=True, help="Path to second dataset")
    compare_parser.add_argument("--name1", default="Dataset1", help="Name for first dataset")
    compare_parser.add_argument("--name2", default="Dataset2", help="Name for second dataset")

    analyze_parser = subparsers.add_parser("analyze-dataset", help="Analyze dataset label distribution")
    analyze_parser.add_argument("--dataset", required=True, help="Path to dataset")

    validate_ds_parser = subparsers.add_parser("validate-dataset", help="Validate dataset structure and format")
    validate_ds_parser.add_argument("--dataset", required=True, help="Path to dataset to validate")

    merge_parser = subparsers.add_parser("merge-datasets", help="Merge two YOLO datasets")
    merge_parser.add_argument("--dataset1", required=True, help="Path to first dataset")
    merge_parser.add_argument("--dataset2", required=True, help="Path to second dataset")
    merge_parser.add_argument("--output", required=True, help="Path to output merged dataset")
    merge_parser.add_argument("--name1", default="dataset1", help="Name prefix for first dataset")
    merge_parser.add_argument("--name2", default="dataset2", help="Name prefix for second dataset")

    subset_parser = subparsers.add_parser("create-subset", help="Create a subset of a dataset")
    subset_parser.add_argument("--dataset", required=True, help="Path to input dataset")
    subset_parser.add_argument("--output", required=True, help="Path to output subset")
    subset_parser.add_argument("--ratio", type=float, default=0.1, help="Subset ratio")
    subset_parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Visualization commands
    visualize_parser = subparsers.add_parser("visualize-obb", help="Visualize oriented bounding boxes")
    visualize_parser.add_argument("--dataset", required=True, help="Path to OBB dataset")
    visualize_parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to visualize",
    )
    visualize_parser.add_argument("--num-samples", type=int, default=6, help="Number of samples to visualize")
    visualize_parser.add_argument("--output", help="Output path for saved visualization")
    visualize_parser.add_argument("--save", action="store_true", help="Save visualization instead of displaying")
    visualize_parser.add_argument("--width", type=int, default=15, help="Figure width")
    visualize_parser.add_argument("--height", type=int, default=10, help="Figure height")

    # Model training and inference commands
    train_parser = subparsers.add_parser("train", help="Train model")
    train_parser.add_argument("--model", default="yolo11n.pt", help="Model path or name")
    train_parser.add_argument("--data", required=True, help="Data path")
    train_parser.add_argument("--task", choices=["detect", "classify"], default="detect", help="Task type")
    train_parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    train_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    train_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    train_parser.add_argument("--device", default="auto", help="Device")
    train_parser.add_argument("--project", default="runs/train", help="Project directory")
    train_parser.add_argument("--name", default="exp", help="Experiment name")
    train_parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    predict_parser = subparsers.add_parser("predict", help="Run prediction")
    predict_parser.add_argument("--model", required=True, help="Model path")
    predict_parser.add_argument("--source", required=True, help="Source path")
    predict_parser.add_argument("--task", choices=["detect", "classify"], default="detect", help="Task type")
    predict_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    predict_parser.add_argument("--device", default="auto", help="Device")
    predict_parser.add_argument("--save", action="store_true", help="Save results")
    predict_parser.add_argument("--project", default="runs/predict", help="Project directory")
    predict_parser.add_argument("--name", default="exp", help="Experiment name")

    validate_parser = subparsers.add_parser("validate", help="Validate model")
    validate_parser.add_argument("--model", required=True, help="Model path")
    validate_parser.add_argument("--data", required=True, help="Data path")
    validate_parser.add_argument("--task", choices=["detect", "classify"], default="detect", help="Task type")
    validate_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    validate_parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    validate_parser.add_argument("--device", default="auto", help="Device")
    validate_parser.add_argument("--project", default="runs/val", help="Project directory")
    validate_parser.add_argument("--name", default="exp", help="Experiment name")

    return parser


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "prepare-data":
            prepare_data_command(args)
        elif args.command == "extract-crops":
            extract_crops_command(args)
        elif args.command == "color-transfer":
            color_transfer_command(args)
        elif args.command == "compare-datasets":
            compare_datasets_command(args)
        elif args.command == "analyze-dataset":
            analyze_dataset_command(args)
        elif args.command == "validate-dataset":
            validate_dataset_command(args)
        elif args.command == "merge-datasets":
            merge_datasets_command(args)
        elif args.command == "create-subset":
            create_subset_command(args)
        elif args.command == "visualize-obb":
            visualize_obb_command(args)
        elif args.command == "train":
            train_command(args)
        elif args.command == "predict":
            predict_command(args)
        elif args.command == "validate":
            validate_model_command(args)
        else:
            parser.print_help()
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
