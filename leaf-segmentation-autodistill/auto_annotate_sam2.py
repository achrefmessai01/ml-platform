from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
import os
import random
import glob

# List of different ontologies to test
ontologies_to_test = [
    CaptionOntology({"every individual leaf": "every individual leaf"}),
]

# Configuration
images_folder = "datasets/merged_data/detection/combined_coco/train"
num_images = 200
output_directory = "output_images/"

def get_random_images(folder_path, num_images):
    """Get random image files from folder"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPEG', '*.JPG', '*.PNG']
    all_images = []
    for ext in image_extensions:
        all_images.extend(glob.glob(os.path.join(folder_path, ext)))
    
    if len(all_images) < num_images:
        print(f"Warning: Only {len(all_images)} images found, using all available")
        return all_images
    
    return random.sample(all_images, num_images)

# Create output directory
os.makedirs(output_directory, exist_ok=True)

# Get random images
image_paths = get_random_images(images_folder, num_images)
print(f"Testing on {len(image_paths)} random images")

# Results storage
results_file = os.path.join(output_directory, "detection_statistics.txt")

with open(results_file, "w") as f:
    f.write("Ontology,Total_Detections,Average_Per_Image\n")
    
    for i, ontology in enumerate(ontologies_to_test):
        prompt_name = ontology.classes()[0]
        print(f"Testing ontology {i+1}/{len(ontologies_to_test)}: {prompt_name}")
        
        base_model = GroundedSAM2(ontology=ontology)
        total_detections = 0
        
        # Test on all selected images
        for img_path in image_paths:
            results = base_model.predict(img_path)
            total_detections += len(results)
        
        average_detections = total_detections / len(image_paths)
        
        print(f"  Total detections: {total_detections}, Average per image: {average_detections:.2f}")
        f.write(f"{prompt_name},{total_detections},{average_detections:.2f}\n")

print(f"Results saved to {results_file}")
