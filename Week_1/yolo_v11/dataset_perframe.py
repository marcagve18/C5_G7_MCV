import os
import cv2
import numpy as np
from pycocotools.mask import decode

# Define class mappings from KITTI-MOTTS to YOLO
category_map = {1: 0, 2: 1, 3: 2}  # YOLO Classes: 0 - Pedestrian, 1 - Car, 2 - Cyclist

def convert_kitti_mots_to_yolo(dataset_path, sequence="0000", frame_to_process="000000"):
    """
    Convert a single frame from KITTI-MOTS to YOLO format and return the image and annotations.

    Args:
        dataset_path (str): Path to KITTI-MOTS dataset.
        sequence (str): Sequence number (default: "0000").
        frame_to_process (str): Frame number (default: "000000").

    Returns:
        tuple: (image (numpy array), annotations (list of strings in YOLO format))
    """

    images_path = os.path.join(dataset_path, "training", "image_02", sequence)
    ann_file = os.path.join(dataset_path, "instances_txt", f"{sequence}.txt")

    # Check if required files exist
    if not os.path.isdir(images_path) or not os.path.isfile(ann_file):
        print(f"Sequence {sequence} or annotation file not found. Exiting.")
        return None, None

    # Read annotation file
    with open(ann_file, "r") as f:
        annotations = f.readlines()

    ann_dict = {}
    for line in annotations:
        parts = line.strip().split(" ")
        if len(parts) < 6:
            continue

        frame, _, class_id, img_height, img_width, rle = parts[:6]
        frame = int(frame)
        class_id = int(class_id)

        if class_id == 10 or class_id not in category_map:
            continue  # Ignore class 10 or unknown classes

        mask = {"counts": rle.strip(), "size": [int(img_height), int(img_width)]}
        bbox = decode(mask).astype(np.uint8)
        x, y, w, h = cv2.boundingRect(bbox)

        # Normalize values for YOLO format
        x_center = (x + w / 2) / int(img_width)
        y_center = (y + h / 2) / int(img_height)
        w = w / int(img_width)
        h = h / int(img_height)

        if frame not in ann_dict:
            ann_dict[frame] = []
        ann_dict[frame].append(f"{category_map[class_id]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

    # Process only the selected frame
    image_path = os.path.join(images_path, f"{frame_to_process}.png")

    if os.path.exists(image_path):
        image = cv2.imread(image_path)  # Load image
        frame_number = int(frame_to_process)  # Convert to integer for comparison
        yolo_annotations = ann_dict.get(frame_number, [])  # Get annotations for this frame

        print(f"Processed frame {frame_to_process} in sequence {sequence}.")
        return image, yolo_annotations

    else:
        print(f"Frame {frame_to_process} not found in sequence {sequence}.")
        return None, None

# Run conversion for only Sequence 0000, Frame 000000
dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
image, annotations = convert_kitti_mots_to_yolo(dataset_path, sequence="0000", frame_to_process="000000")