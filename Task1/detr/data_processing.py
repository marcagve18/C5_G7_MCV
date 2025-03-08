import pycocotools.mask
import os
import pycocotools
from PIL import Image
from transformers import AutoImageProcessor
from tqdm import tqdm 
from datasets import Dataset
import torchvision
import json
import tempfile
import torchvision
import numpy as np

dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"


def get_hf_dataset():

    image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")

    kitti_coco_dataset = []
    for instance_metadata in sorted(os.listdir(dataset_path + "/instances_txt")):
        instance_number = int(instance_metadata.split(".")[0])
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

                bbox = pycocotools.mask.toBbox(coco_decode)
                area = pycocotools.mask.area(coco_decode)

                image_id = int(str(instance_number) + "{:06d}".format(int(time_frame)))

                coco_format = {
                    "image_id": image_id,
                    "category_id": class_id,
                    "iscrowd": 0,
                    "area": area,
                    "bbox": list(bbox)
                }

                if image_id not in time_frames_annotations.keys():
                    time_frames_annotations[image_id] = [coco_format]
                else:
                    time_frames_annotations[image_id].append(coco_format)
            
            for img_id in time_frames_annotations.keys():
                kitti_coco_dataset.append({
                    "image_id": img_id,
                    "annotations": time_frames_annotations[img_id]
                })


    processed_instances = []
    images_path = os.path.join(dataset_path, 'training/image_02')
    for instance_id in tqdm(sorted(os.listdir(images_path))):
        print("instance id: ", instance_id)
        instance_path = os.path.join(images_path, instance_id)
        img_paths = sorted(os.listdir(instance_path))
        img_batch = [Image.open(os.path.join(instance_path, img)) for img in img_paths]
        for img_idx, img in enumerate(img_batch):
            processed_image = image_processor(images=img, annotations=kitti_coco_dataset[img_idx],  return_tensors="pt")
            processed_instances.append(processed_image)
        print(instance_id)
    

    print("Creating HF dataset")
    hf_dataset = Dataset.from_dict({
            "pixel_values": [instance["pixel_values"] for instance in processed_instances],
            "pixel_mask": [instance["pixel_mask"] for instance in processed_instances],
            "labels": [instance["labels"] for instance in processed_instances],
    })
    print("HF Dataset created")
        
    return hf_dataset


class KITTIMOTS_CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, dataset_path, processor_checkpoint="facebook/detr-resnet-50", instances_ids = None, transforms=None):
        """
        Args:
            dataset_path (str): Path to the KITTI-MOTS dataset.
            processor_checkpoint (str): Checkpoint for the image processor.
            transforms (callable, optional): Any additional transforms.
        """
        self.dataset_path = dataset_path
        self.image_processor = AutoImageProcessor.from_pretrained(processor_checkpoint)
        self.albumentations_transforms = transforms
        
        # Build a COCO-formatted annotations dictionary from the KITTI-MOTS annotation files.
        coco_ann = self.build_coco_annotations(dataset_path, instances_ids)

        # Write the COCO annotations to a temporary file.
        tmp_ann_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json")
        json.dump(coco_ann, tmp_ann_file)
        tmp_ann_file.close()
        ann_file = tmp_ann_file.name
        
        # The images are stored in training/image_02 with instance folders.
        images_folder = os.path.join(dataset_path, "training", "image_02")
        # Initialize the parent CocoDetection dataset.
        super().__init__(images_folder, ann_file, transforms=None)
    
    def build_coco_annotations(self, dataset_path, instances_ids=None):
        """
        Parse KITTI-MOTS annotation files and images to build a COCO-formatted dictionary.
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
           
            if instances_ids: # To properly select train and test splits we filter instances if specified
                if instance_number not in instances_ids:
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
                    
                    # Decode the mask to get bbox and area.
                    coco_decode = {"counts": rle, "size": [img_height, img_width]}
                    bbox = pycocotools.mask.toBbox(coco_decode).tolist()
                    area = float(pycocotools.mask.area(coco_decode))
                    
                    # Create a unique image ID (combining instance number and time frame).
                    image_id = int(str(instance_number) + "{:06d}".format(int(time_frame)))
                    
                    category_map = {
                        1: 3, #car
                        2: 1, #pedestrian
                    }

                    if class_id == 10:
                        continue

                    ann = {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_map[class_id],
                        "iscrowd": 0,
                        "area": area,
                        "bbox": bbox
                    }
                    annotation_id += 1
                    annotations_dict.setdefault(image_id, []).append(ann)
                    categories_set.add(category_map[class_id])
        
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
            if instances_ids: # To properly select train and test splits we filter instances if specified
                if instance_number not in instances_ids:
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
                    "file_name": file_path  # absolute path is fine
                })
        
        # Flatten the annotations.
        annotations = []
        for anns in annotations_dict.values():
            annotations.extend(anns)
        
        categories_names = {
            1: "pedestrian",
            3: "car",
        }
        # Build categories list.
        categories = [{"id": cat_id, "name": categories_names[cat_id]} for cat_id in sorted(categories_set)]
    
        # Final COCO dictionary.
        coco_dict = {"images": images, "annotations": annotations, "categories": categories}
        return coco_dict

    def __getitem__(self, index):
        # Get the raw image and target (in COCO format) from the parent class.
        img, target = super().__getitem__(index)
        
        # If albumentations transforms are provided, apply them.
        if self.albumentations_transforms:
            # Convert the PIL image to a NumPy array.
            image_np = np.array(img)
            
            # Extract bounding boxes and labels from target annotations.
            bboxes = [ann["bbox"] for ann in target]
            labels = [ann["category_id"] for ann in target]
            print(labels)
            augmented = self.albumentations_transforms(
                image=image_np,
                bboxes=bboxes,
                category=labels
            )
            # Update the image and bounding boxes with the transformed results.
            image_np = augmented["image"]
            new_bboxes = augmented["bboxes"]
            
            for ann, new_bbox in zip(target, new_bboxes):
                ann["bbox"] = new_bbox
            
            img = image_np
        
        # Get the image_id and create a target dictionary.
        image_id = self.ids[index]
        target_dict = {"image_id": image_id, "annotations": target}
        
        # Process image and annotations with the image processor.
        encoding = self.image_processor(images=img, annotations=target_dict, return_tensors="pt")
        
        # Remove the batch dimension (processing one image at a time).
        return encoding

        
