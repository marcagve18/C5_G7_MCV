import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO("yolo11m-seg.pt")

# Read input image
image_path = "/home/mcv/datasets/C5/KITTI-MOTS/training/image_02/0019/000000.png"
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Failed to load image: {image_path}")

# Perform inference
results = model(image)

# Define allowed classes (car = 2, person = 0 in COCO dataset)
allowed_classes = ["car", "person"]

# Process results
for result in results:
    masks = result.masks  # Segmentation masks
    boxes = result.boxes  # Bounding boxes
    names = model.names   # Class names

    if masks is not None:
        for i, mask in enumerate(masks.data):
            class_id = int(result.boxes.cls[i])  # Get class ID
            class_name = names[class_id]  # Get class name

            if class_name not in allowed_classes:
                continue  # Skip if not a car or person

            # Convert mask to NumPy format and resize
            mask = mask.cpu().numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

            # Convert to uint8 format and create a colored overlay
            mask = (mask * 255).astype(np.uint8)
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  # Color masks

            # Apply transparency to mask overlay
            alpha = 0.5
            image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

        # Draw bounding boxes and labels
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = names[class_id]

            if class_name not in allowed_classes:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0] * 100  # Confidence score

            label = f"{class_name} {conf:.0f}%"

            # Choose color (Green for person, Blue for car)
            color = (0, 255, 0) if class_name == "person" else (255, 0, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_x1, label_y1 = x1, y1 - label_size[1] - 5
            label_x2, label_y2 = x1 + label_size[0] + 5, y1
            cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), color, -1)

            # Put text on the image
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

# Fix blue tint by converting BGR to RGB
#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Save the output image
output_path = "output.jpg"
cv2.imwrite(output_path, image)

# Display using matplotlib
plt.imshow(image)
plt.axis("off")
plt.show()

print(f"Segmented image saved at: {output_path}")
