import os
import cv2
import numpy as np
from pathlib import Path

# Define directories
image_dir = Path('/ghome/c5mcv07/mcv/datasets/C5/KITTI-MOTS/training/image_02')
mask_dir = Path('/ghome/c5mcv07/mcv/datasets/C5/KITTI-MOTS/instances')
label_dir = Path('/home/c5mcv07/C5_G7_MCV/Week_2/yolo/labels/train')

# Class ID mappings (you can add more classes as per your dataset)
class_mapping = {1: "car", 2: "pedestrian"}  # Adjust based on your dataset

# Helper function to get bounding boxes from mask
def get_bounding_boxes_from_mask(mask):
    # Convert mask to binary (assuming masks are grayscale)
    _, binary_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    
    # Find contours (which represent object boundaries)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes from contours
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, x + w, y + h))
    
    return boxes

# Function to convert bounding box to YOLO format
def convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height):
    # YOLO format: class_id, x_center, y_center, width, height (normalized)
    x_center = (x_min + x_max) / 2 / img_width
    y_center = (y_min + y_max) / 2 / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return x_center, y_center, width, height

# Process each sequence in the dataset
for seq in os.listdir(image_dir):
    seq_image_dir = image_dir / seq
    seq_mask_dir = mask_dir / seq
    seq_label_dir = label_dir / seq
    seq_label_dir.mkdir(parents=True, exist_ok=True)  # Create label directory if not exists
    
    # Process each frame in the sequence
    for frame_num in os.listdir(seq_image_dir):
        frame_img_path = seq_image_dir / frame_num
        frame_mask_path = seq_mask_dir / frame_num
        
        # Read the frame image and mask
        frame_img = cv2.imread(str(frame_img_path))
        mask = cv2.imread(str(frame_mask_path), cv2.IMREAD_UNCHANGED)
        
        # Get image dimensions
        img_height, img_width = frame_img.shape[:2]
        
        # Extract bounding boxes from the mask
        boxes = get_bounding_boxes_from_mask(mask)
        
        # Prepare YOLO labels for this frame
        label_file_path = seq_label_dir / f"{frame_num.replace('.png', '.txt')}"
        
        with open(label_file_path, 'w') as label_file:
            for box in boxes:
                # Assuming the class IDs are encoded in the mask values (e.g., car=1001, pedestrian=2001)
                obj_id = np.unique(mask[box[1]:box[3], box[0]:box[2]])[0]  # Get the unique object id in the bounding box
                class_id = obj_id // 1000  # Class ID (floor division by 1000)
                
                # Convert bounding box to YOLO format
                x_min, y_min, x_max, y_max = box
                x_center, y_center, width, height = convert_to_yolo_format(x_min, y_min, x_max, y_max, img_width, img_height)
                
                # Write to YOLO label file
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

        print(f"Processed {frame_img_path}")
