#!/usr/bin/env python3
"""
YOLO11 Validation Script for Strawberry Classification
"""

from ultralytics import YOLO
import argparse
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Validate YOLO11 model for strawberry detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (e.g., runs/train/strawberry_classifier/weights/best.pt)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (will look for data.yaml in data/processed/{dataset}/data.yaml)')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data configuration file (overrides --dataset)')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size for validation')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for validation')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (cpu, 0, 1, etc.)')
    parser.add_argument('--project', type=str, default='runs/val',
                       help='Project directory to save results')
    parser.add_argument('--name', type=str, default='strawberry_classifier',
                       help='Experiment name')
    
    args = parser.parse_args()
    
    # Resolve data config path
    if args.data:
        data_config = args.data
    else:
        # Build path to dataset's data.yaml
        data_config = f"data/processed/{args.dataset}/data.yaml"
    
    print(f"Using data config: {data_config}")
    
    # Load trained model
    model = YOLO(args.model)
    
    # Validate the model
    results = model.val(
        data=data_config,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        save_json=True,
        save_hybrid=True
    )
    
    print(f"Validation completed! Results saved to: {results.save_dir}")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")

if __name__ == "__main__":
    main()
