import os
import json
import argparse
from PIL import Image
import pycocotools.mask
import numpy as np

def build_coco_annotations(dataset_path, instances_ids=None):
    """
    Parse KITTI-MOTS annotation files and images to build a COCO-formatted dictionary.
    
    Args:
        dataset_path (str): Path to the KITTI-MOTS dataset.
        instances_ids (list, optional): List of instance ids to include. If None, include all.
        
    Returns:
        dict: COCO-formatted annotations dictionary.
    """
    ann_dir = os.path.join(dataset_path, "instances_txt")
    annotations_dict = {}
    annotation_id = 1
    categories_set = set()

    # Process each annotation file.
    for ann_file in sorted(os.listdir(ann_dir)):
        try:
            instance_number = int(ann_file.split(".")[0])
        except ValueError:
            continue

        if instances_ids and instance_number not in instances_ids:
            continue

        file_path = os.path.join(ann_dir, ann_file)
        with open(file_path, "r") as file:
            for line in file:
                parts = line.strip().split(" ")
                if len(parts) < 6:
                    continue
                time_frame = parts[0]
                # parts[1] is object_id (unused here)
                class_id = int(parts[2])
                img_height = int(parts[3])
                img_width = int(parts[4])
                rle = parts[5].strip()

                # Skip class id 10 as per your logic.
                if class_id == 10:
                    continue

                # Build a COCO-compatible mask decoding.
                coco_decode = {"counts": rle, "size": [img_height, img_width]}
                bbox = pycocotools.mask.toBbox(coco_decode).tolist()
                area = float(pycocotools.mask.area(coco_decode))

                # Create a unique image ID (combining instance number and time frame).
                image_id = int(str(instance_number) + "{:06d}".format(int(time_frame)))
                
                

                mask_binary = pycocotools.mask.decode(coco_decode)
                mask_rle = pycocotools.mask.encode(np.asfortranarray(mask_binary))
                mask_rle["counts"] = mask_rle["counts"].decode("utf-8") 
                
                ann = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": bbox,
                    "segmentation": mask_rle
                }
                annotation_id += 1
                annotations_dict.setdefault(image_id, []).append(ann)
                categories_set.add(class_id)
    
    # Build the images list by scanning through the images folder.
    images = []
    images_folder = os.path.join(dataset_path, "training", "image_02")
    for instance_dir in sorted(os.listdir(images_folder)):
        instance_path = os.path.join(images_folder, instance_dir)
        if not os.path.isdir(instance_path):
            continue
        try:
            instance_number = int(instance_dir)
        except ValueError:
            continue

        if instances_ids and instance_number not in instances_ids:
            continue

        for img_file in sorted(os.listdir(instance_path)):
            # Assume file name is like "000001.png" – extract the time frame.
            time_frame_str = os.path.splitext(img_file)[0]
            try:
                time_frame = int(time_frame_str)
            except ValueError:
                continue
            image_id = int(str(instance_number) + "{:06d}".format(time_frame))
            file_path = os.path.join(instance_path, img_file)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
            except Exception:
                width, height = 0, 0
            images.append({
                "id": image_id,
                "width": width,
                "height": height,
                "file_name": file_path  # Using absolute path (or relative if preferred)
            })
    
    # Flatten the annotations.
    annotations = []
    for anns in annotations_dict.values():
        annotations.extend(anns)
    
    categories_names = {
        1: "car",
        2: "pedestrian",
    }
    categories = [{"id": cat_id, "name": categories_names[cat_id]} for cat_id in sorted(categories_set)]
    
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    return coco_dict

def main():
    dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
    test_instances_ids = [10, 6, 2, 16, 0, 17, 3, 14, 12]


    coco_annotations = build_coco_annotations(dataset_path, test_instances_ids)
    
    with open("test_annotations.json", "w") as f:
        json.dump(coco_annotations, f)
    print(f"COCO annotations saved to {"test_annotations.json"}")

if __name__ == "__main__":
    main()
