from rfdetr import RFDETRLarge
import numpy as np
import supervision as sv
from PIL import Image

weights_path = "checkpoint_best_total.pth"
image_path = "dataset1_5d233ff4-IMG_9336_jpg.rf.62daba4b2dce52c85189137c341b73d5.jpg"
model = RFDETRLarge(pretrain_weights=weights_path)
model.optimize_for_inference(compile=False)

img = Image.open(image_path)
detections = model.predict(img, threshold=0.5)

print("Boxes:", detections.xyxy)
print("Scores:", detections.confidence)
print("Classes:", detections.class_id)

# Visualize detections using SuoVision
annotated_image = img.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections)

# Save the annotated image
annotated_image.save("detection_result.jpg")
print("Detection result saved as 'detection_result.jpg'")