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
    parser = argparse.ArgumentParser(description='Train YOLO11 model for strawberry detection or classification')
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                       help='Model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--data', type=str, default='data/processed/farm_dataset/data.yaml',
                       help='Path to data configuration file (detection) or dataset directory (classification)')
    parser.add_argument('--task', type=str, choices=['detect', 'classify'], default='detect',
                       help='Training task: detect (object detection) or classify (image classification)')
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
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (number of epochs with no improvement)')
    
    args = parser.parse_args()
    
    # Ensure local cache directory exists
    cache_dir = PROJECT_ROOT / 'models' / 'cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine model type based on task
    if args.task == 'classify':
        # For classification, use YOLO11 classification models
        if args.model.endswith('.pt') and 'cls' not in args.model:
            # Convert detection model name to classification model name
            model_base = args.model.replace('.pt', '')
            model_name = f"{model_base}-cls.pt"
            print(f"🔄 Converting detection model '{args.model}' to classification model '{model_name}'")
        else:
            model_name = args.model
        
        # Validate classification dataset structure
        data_path = Path(args.data)
        if not data_path.exists():
            raise ValueError(f"Classification dataset directory not found: {args.data}")
        
        # Check for expected classification structure (train/val directories with class subdirs)
        expected_dirs = ['train', 'val']
        missing_dirs = [d for d in expected_dirs if not (data_path / d).exists()]
        if missing_dirs:
            print(f"⚠️  Missing directories in classification dataset: {missing_dirs}")
            print(f"📁 Expected structure: {args.data}/{{train,val}}/{{class1,class2,...}}")
        
        print(f"📊 Classification dataset: {args.data}")
        
    else:  # detect
        model_name = args.model
        
        # Validate detection dataset (data.yaml should exist)
        if not args.data.endswith('.yaml') and not args.data.endswith('.yml'):
            print(f"⚠️  Detection data path should point to a .yaml file, got: {args.data}")
        
        data_path = Path(args.data)
        if not data_path.exists():
            raise ValueError(f"Detection data configuration file not found: {args.data}")
        
        print(f"🎯 Detection dataset config: {args.data}")
    
    # Load YOLO11 model (will download to local cache if needed)
    print(f"Loading model: {model_name}")
    model = YOLO(model_name)
    
    # Enable TensorBoard logging if requested
    if args.tensorboard:
        try:
            import tensorboard  # noqa: F401
            print("📈 TensorBoard logging enabled - logs will be saved automatically")
        except ImportError:
            print("⚠️  TensorBoard not found, installing...")
            subprocess.run([sys.executable, "-m", "pip", "install", "tensorboard"], check=True)
            print("✅ TensorBoard installed successfully!")
    
    # Prepare training arguments based on task
    train_kwargs = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch_size,
        'imgsz': args.img_size,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'patience': args.patience,
        'save': True,
        'verbose': True,
        # 'freeze': 9
    }
    
    # Add task-specific parameters
    if args.task == 'classify':
        # Classification-specific parameters
        train_kwargs.update({
            'save_period': 10,  # Save checkpoint every 10 epochs
            'dropout': 0.0,     # Dropout rate for classification
        })
        print(f"🚀 Starting YOLO11 classification training...")
    else:  # detect
        # Detection-specific parameters
        train_kwargs.update({
            'save_period': 10,  # Save checkpoint every 10 epochs
        })
        print(f"🚀 Starting YOLO11 detection training...")
    
    # Train the model
    results = model.train(**train_kwargs)
    
    # Print completion message with task-specific info
    save_dir = Path(args.project) / args.name
    print(f"✅ {args.task.capitalize()} training completed! Results saved to: {save_dir}")
    
    # Print task-specific results summary
    if args.task == 'classify':
        print("📊 Classification Results:")
        print(f"  - Model: {model_name}")
        print(f"  - Dataset: {args.data}")
        print(f"  - Final weights: {save_dir / 'weights' / 'best.pt'}")
    else:
        print("🎯 Detection Results:")
        print(f"  - Model: {model_name}")
        print(f"  - Dataset config: {args.data}")
        print(f"  - Final weights: {save_dir / 'weights' / 'best.pt'}")
    
    return results

if __name__ == "__main__":
    main()
