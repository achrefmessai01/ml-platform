#!/usr/bin/env python3
"""
YOLO11 Training Script for Strawberry Classification
"""

from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='Train YOLO11 model for strawberry detection')
    parser.add_argument('--model', type=str, default='yolo11n.pt', 
                       help='Model variant (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt)')
    parser.add_argument('--data', type=str, default='configs/data.yaml',
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
    
    args = parser.parse_args()
    
    # Load YOLO11 model
    model = YOLO(args.model)
    
    # Train the model
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
    
    print(f"Training completed! Results saved to: {results.save_dir}")

if __name__ == "__main__":
    main()
