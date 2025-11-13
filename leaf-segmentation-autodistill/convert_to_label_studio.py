#!/usr/bin/env python3

import os
import json
from pathlib import Path

def convert_yolo_to_label_studio(annotations_dir, output_dir="labels"):
    """Convert YOLO segmentation annotations to Label Studio format."""
    
    os.makedirs(output_dir, exist_ok=True)
    annotation_files = list(Path(annotations_dir).glob("*.txt"))
    
    tasks = []
    
    for idx, annotation_file in enumerate(annotation_files):
        image_filename = annotation_file.stem + ".jpg"
        s3_url = f"s3://datasets/merged_data/all_images_labeled/images/{image_filename}"
        
        # Parse YOLO annotations
        objects = []
        with open(annotation_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue
                    
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                
                # Convert to percentage coordinates for Label Studio
                points = []
                for i in range(0, len(coords), 2):
                    x = coords[i] * 100
                    y = coords[i + 1] * 100
                    points.append([x, y])
                
                objects.append({
                    "original_width": 1920,
                    "original_height": 1080,
                    "image_rotation": 0,
                    "value": {
                        "points": points,
                        "polygonlabels": ["leaf"]  # Default to "leaf" class
                    },
                    "id": f"seg{idx}_{len(objects)}",
                    "from_name": "label",
                    "to_name": "image",
                    "type": "polygonlabels"
                })
        
        if objects:
            task = {
                "data": {"image": s3_url},
                "annotations": [{
                    "id": idx + 1,
                    "created_username": "admin",
                    "created_ago": "0 minutes",
                    "task": idx + 1,
                    "result": objects
                }]
            }
            
            tasks.append(task)
            
            # Save individual file
            with open(f"{output_dir}/{annotation_file.stem}.json", 'w') as f:
                json.dump(task, f, indent=2)
    
    # Save all tasks
    with open(f"{output_dir}/all_tasks.json", 'w') as f:
        json.dump(tasks, f, indent=2)
    
    print(f"Converted {len(tasks)} files to {output_dir}/")

if __name__ == "__main__":
    import sys
    annotations_dir = sys.argv[1] if len(sys.argv) > 1 else "datasets/merged_data/all_images_labeled/annotations"
    convert_yolo_to_label_studio(annotations_dir)