#!/usr/bin/env python3
"""
YOLO11 Prediction Script for Strawberry Classification
"""

import os
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run inference with YOLO11 model for strawberry detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (e.g., runs/train/strawberry_classifier/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True,
                       help='Path to image, video, directory, or URL')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for NMS')
    parser.add_argument('--img-size', type=int, default=640,
                       help='Image size for inference')
    parser.add_argument('--device', type=str, default='',
                       help='Device to use (cpu, 0, 1, etc.)')
    parser.add_argument('--project', type=str, default='runs/predict',
                       help='Project directory to save results')
    parser.add_argument('--name', type=str, default='strawberry_classifier',
                       help='Experiment name')
    parser.add_argument('--save', action='store_true',
                       help='Save results to file')
    parser.add_argument('--save-txt', action='store_true',
                       help='Save results as txt files')
    parser.add_argument('--save-conf', action='store_true',
                       help='Save confidence scores in txt files')
    
    args = parser.parse_args()
    
    # Load trained model
    model = YOLO(args.model)
    
    # Run inference
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.img_size,
        device=args.device,
        project=args.project,
        name=args.name,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        show=False,
        verbose=True
    )
    
    print(f"Inference completed! Results saved to: {results[0].save_dir if results else 'No results'}")

if __name__ == "__main__":
    main()
