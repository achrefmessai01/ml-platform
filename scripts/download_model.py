#!/usr/bin/env python3
"""
Download YOLO11 pre-trained models
"""

import os
import requests
from pathlib import Path

def download_model(model_name: str = "yolo11n.pt", models_dir: str = "models"):
    """
    Download YOLO11 pre-trained model
    
    Args:
        model_name: Name of the model to download (yolo11n.pt, yolo11s.pt, etc.)
        models_dir: Directory to save the model
    """
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, model_name)
    
    # Check if model already exists
    if os.path.exists(model_path):
        print(f"Model {model_name} already exists at {model_path}")
        return model_path
    
    # YOLO11 models will be downloaded automatically when first used
    # This script serves as a placeholder for manual downloads if needed
    print(f"YOLO11 models are downloaded automatically when first used.")
    print(f"When you run training, {model_name} will be downloaded to {model_path}")
    
    return model_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download YOLO11 pre-trained models')
    parser.add_argument('--model', type=str, default='yolo11n.pt',
                       choices=['yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt'],
                       help='Model to download')
    parser.add_argument('--models-dir', type=str, default='models',
                       help='Directory to save models')
    
    args = parser.parse_args()
    download_model(args.model, args.models_dir)
