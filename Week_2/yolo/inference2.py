import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
from ultralytics import YOLO

# Load the YOLOv8 segmentation model
model = YOLO("yolo_finetuned_model.pt")

# Read input image
image_path = "/home/mcv/datasets/C5/KITTI-MOTS/training/image_02/0019/000000.png"
image = cv2.imread(image_path)
if image is None:
    raise ValueError(f"Failed to load image: {image_path}")

# Start measuring inference time
start_time = time.time()

# Perform inference
results = model(image)

# End inference time measurement
end_time = time.time()
inference_time = end_time - start_time
print(f"Inference Time for single image: {inference_time:.4f} seconds")

# Define allowed classes (car = 2, person = 0 in COCO dataset)
allowed_classes = ["car", "person"]

# Initialize an empty list to store segmented images
segmented_images = []

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

            # Convert mask to NumPy format and resize to match the original image size
            mask = mask.cpu().numpy()
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

            # Convert mask to uint8 format
            mask = (mask * 255).astype(np.uint8)

            # Create a black background image to place the segmented object
            segmented_image = np.zeros_like(image)

            # Use the mask to extract the segmented object from the original image
            segmented_image[mask == 255] = image[mask == 255]

            # Store the segmented image (isolated object)
            segmented_images.append(segmented_image)

            # Draw bounding boxes and labels on the original image
            x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])  # Get bounding box coordinates
            conf = boxes[i].conf[0] * 100  # Confidence score

            label = f"{class_name} {conf:.0f}%"

            # Choose color (Green for person, Blue for car)
            color = (0, 255, 0) if class_name == "person" else (255, 0, 0)

            # Draw the bounding box on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            label_x1, label_y1 = x1, y1 - label_size[1] - 5
            label_x2, label_y2 = x1 + label_size[0] + 5, y1
            cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), color, -1)

            # Put text on the image
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Overlay the segmentation mask in a colored way
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)  # Color masks
            alpha = 0.5
            image = cv2.addWeighted(image, 1, colored_mask, alpha, 0)

# Save the full image with bounding boxes, labels, and segmentation masks
output_image_path = "output_with_bboxes_and_masks.jpg"
cv2.imwrite(output_image_path, image)

print(f"Image with bounding boxes and masks saved at: {output_image_path}")

# Save the segmented (isolated) objects as separate images
for idx, segmented_image in enumerate(segmented_images):
    segmented_output_path = f"segmented_object_{idx}.png"
    cv2.imwrite(segmented_output_path, segmented_image)
    print(f"Segmented object saved at: {segmented_output_path}")

# Optionally, display the final image with bounding boxes and segmentation masks
plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
