import pycocotools.mask
import os
import pycocotools
import numpy as np
import cv2
from tqdm import tqdm

dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
output_dataset_path = "/ghome/c5mcv07/C5_G7_MCV/Week_2/huggingface/processed_datasets"

os.makedirs(os.path.join(output_dataset_path, 'KITTI_MOTS'), exist_ok=True)

for instance_metadata in tqdm(sorted(os.listdir(dataset_path + "/instances_txt"))):
    instance_str = instance_metadata.split(".")[0]
    instance_number = int(instance_str)
    masks = {}
    images = {}
    with open(dataset_path + "/instances_txt/" + instance_metadata, "r") as file:
        time_frames_annotations = {}
        for line in file:
            splitted = line.split(" ")
            time_frame = splitted[0]
            object_id = splitted[1]
            class_id = splitted[2]
            img_height = splitted[3]
            img_width = splitted[4]
            rle = splitted[5].replace("\n", "")

            coco_decode = {
                "counts": rle,
                "size": [int(img_height), int(img_width)]
            }

            mask = pycocotools.mask.decode(coco_decode)

            if time_frame not in masks:
                masks[time_frame] = np.zeros(mask.shape)

            mask_label_encoded = mask * int(class_id)
            masks[time_frame] += mask_label_encoded

            if time_frame not in images:
                images[time_frame] = cv2.imread(os.path.join(dataset_path, f'training/image_02/{instance_str}/{int(time_frame):06d}.png'))

    output_instance_path = os.path.join(output_dataset_path, f'KITTI_MOTS/{instance_str}')
    os.makedirs(output_instance_path, exist_ok=True)
    for (time_frame, image), (_, mask_annotation) in zip(images.items(), masks.items()):
        # Optionally, assert that the keys match:
        assert time_frame == _, "Keys do not match"
        cv2.imwrite(f"{os.path.join(output_instance_path, f"{int(time_frame):06d}")}.png", image)
        cv2.imwrite(f"{os.path.join(output_instance_path, f"{int(time_frame):06d}")}_mask.png", mask_annotation)



 