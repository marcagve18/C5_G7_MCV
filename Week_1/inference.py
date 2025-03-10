from PIL import Image, ImageDraw
from ultralytics import YOLO
import cv2
import torch
import matplotlib.pyplot as plt
from dataset_perframe import convert_kitti_mots_to_yolo


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = YOLO("yolo11n.pt")


dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
image, annotations = convert_kitti_mots_to_yolo(dataset_path, sequence="0000", frame_to_process="000000")
# Confidence threshold
confidence_threshold = 0.7  # You can adjust this value (e.g., 0.5 means 50%)

# Run inference
results = model(image)

# Get the predictions (boxes, labels, and confidences)
predictions = results[0].boxes  # The first result (since batch size is 1)
boxes = predictions.xywh  # Bounding boxes (center_x, center_y, width, height)
labels = predictions.cls  # Class labels
confidences = predictions.conf  # Confidence scores

# Get class names (if using COCO)
class_names = results[0].names  # Class names from COCO dataset

# Classes to filter (only 'person' and 'car')
target_classes = ['person', 'car']

# Draw the bounding boxes on the image, only if confidence > threshold
for i, box in enumerate(boxes):
    # Extract the box coordinates
    x1, y1, w, h = box
    x1, y1 = int(x1 - w / 2), int(y1 - h / 2)  # Convert center to top-left corner
    x2, y2 = int(x1 + w), int(y1 + h)  # Bottom-right corner

    # Get label and confidence
    label = class_names[int(labels[i])]
    confidence = confidences[i]

    # Only draw the box if the confidence is above the threshold
    if confidence > confidence_threshold and label in target_classes:
        # Draw the rectangle (bounding box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        # Add label and confidence text
        cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Display the result


# Save the result with bounding boxes
cv2.imwrite("output_with_boxes_threshold.jpg", image)
print("Inference completed! Saved result as 'output_with_boxes_threshold.jpg'.")