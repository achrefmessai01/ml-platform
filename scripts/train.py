#!/usr/bin/env python3
"""
YOLO11 Training Script for Strawberry Classification
"""

from ultralytics import YOLO
import argparse
import os
import sys
import subprocess
from pathlib import Path

# Set local cache directories to avoid global downloads
PROJECT_ROOT = Path(__file__).parent.parent
os.environ['TORCH_HOME'] = str(PROJECT_ROOT / 'models' / 'cache')

def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 model for strawberry detection')
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                       help='Model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--data', type=str, default='data/processed/farm_dataset/data.yaml',
                       help='Path to data configuration file')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for training')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (cpu, 0, 1, etc.)')
    parser.add_argument('--project', type=str, default='runs/train',
                       help='Project directory to save results')
    parser.add_argument('--name', type=str, default='strawberry_classifier',
                       help='Experiment name')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Enable TensorBoard logging during training')
    
    args = parser.parse_args()
    
    # Ensure local cache directory exists
    cache_dir = PROJECT_ROOT / 'models' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Load YOLO11 model (will download to local cache if needed)
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    # Enable TensorBoard logging if requested
    if args.tensorboard:
        try:
            import tensorboard  # noqa: F401
            print("📈 TensorBoard logging enabled - logs will be saved automatically")
        except ImportError:
            print("⚠️  TensorBoard not found, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"], check=True)
            print("✅ TensorBoard installed successfully!")
    
    # Train the model (TensorBoard logging is automatic when tensorboard is available)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        verbose=True
    )
    
    # Print completion message
    save_dir = Path(args.project) / args.name
    print(f"Training completed! Results saved to: {save_dir}")

if __name__ == "__main__":
    main()
