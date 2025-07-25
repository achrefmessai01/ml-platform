#!/usr/bin/env python3
"""
YOLO11 Training Script for Strawberry Classification
"""

from ultralytics import YOLO
import argparse
import os
import sys
import subprocess
import yaml
from pathlib import Path

# Set local cache directories to avoid global downloads
PROJECT_ROOT = Path(__file__).parent.parent
os.environ['TORCH_HOME'] = str(PROJECT_ROOT / 'models' / 'cache')

def load_config(config_path):
    """Load training configuration from a YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"📄 Loaded configuration from: {config_path}")
        return config
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing configuration file: {e}")

def merge_config_with_args(config, args):
    """Merge configuration file parameters with command line arguments.
    Command line arguments take precedence over config file parameters.
    """
    # Convert args namespace to dict
    args_dict = vars(args)
    
    # Start with config parameters
    merged_config = config.copy() if config else {}
    
    # Override with command line arguments (only if they were explicitly provided)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to training configuration file (YAML)')
    parser.add_argument('--model', type=str, default='yolo11n.pt')
    parser.add_argument('--data', type=str, default='data/processed/farm_dataset/data.yaml')
    parser.add_argument('--task', type=str, choices=['detect', 'classify'], default='detect')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default='strawberry_classifier')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--single-cls', type=bool, default=False)
    parser.add_argument('--resume', type=bool, default=False)
    
    # Get defaults
    defaults = vars(parser.parse_args([]))
    
    # Update merged config with non-default command line arguments
    for key, value in args_dict.items():
        if key != 'config' and (key not in defaults or value != defaults[key]):
            # Convert underscores to hyphens for consistency with config file format
            config_key = key.replace('_', '-')
            merged_config[config_key] = value
    
    # Handle any extra parameters from config that aren't in the argument parser
    # These will be passed directly to YOLO training
    extra_params = {}
    valid_args = set(defaults.keys())
    # Create all valid argument variations (with underscores and hyphens)
    valid_args_expanded = set(valid_args)
    for key in list(valid_args):
        valid_args_expanded.add(key.replace('-', '_'))
        valid_args_expanded.add(key.replace('_', '-'))
    
    for key, value in merged_config.items():
        normalized_key = key.replace('-', '_')
        if normalized_key not in valid_args_expanded:
            extra_params[key] = value
    
    return merged_config, extra_params

def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 model for strawberry detection or classification')
    parser.add_argument('--config', type=str, help='Path to training configuration file (YAML)')
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
    parser.add_argument('--single-cls', type=bool, default=False,
                       help='Use single class training (for classification tasks)')
    parser.add_argument('--resume', type=bool, default=False,
                       help='Resume training from last checkpoint if available')
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config = None
    if args.config:
        config = load_config(args.config)
    
    # Merge configuration with command line arguments
    merged_config, extra_params = merge_config_with_args(config, args)
    
    # Convert merged config back to namespace for compatibility
    for key, value in merged_config.items():
        # Convert hyphen keys to underscore attributes
        attr_name = key.replace('-', '_')
        if hasattr(args, attr_name):
            setattr(args, attr_name, value)
    
    if config and args.config:
        print(f"✅ Configuration loaded and merged. Extra YOLO parameters: {list(extra_params.keys()) if extra_params else 'None'}")
    
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
        'single_cls': args.single_cls,
        'resume': args.resume,  # Resume training if last checkpoint exists
        'save': True,
        'verbose': True,
        'workers': 4,  # Reduce number of workers to prevent segmentation fault
        # 'freeze': 9
    }
    
    # Add extra parameters from configuration file
    if extra_params:
        print(f"🔧 Adding extra parameters from config: {extra_params}")
        train_kwargs.update(extra_params)
    
    # TensorBoard logging info
    if args.tensorboard:
        print(f"📈 TensorBoard logging: YOLO11 will automatically create logs if TensorBoard is installed")
        print(f"� Logs will be saved to: {args.project}/{args.name}/")
        print("💡 Run 'tensorboard --logdir=runs/train' to view logs after training starts")
    
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
