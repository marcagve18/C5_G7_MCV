from datasets import Dataset, DatasetDict
from datasets import Image as DatasetImage
from PIL import Image
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv('HF_MARC')
login(HF_TOKEN)

train_instances_ids = [19, 20, 9, 7, 1, 8, 15, 11, 13, 18, 4, 5]
test_instances_ids = [10, 6, 2, 16, 0, 17, 3, 14, 12]

label2id = {
    "background": 0,
    "pedestrian": 2,
    "car": 1,
}

dataset_path = "/ghome/c5mcv07/mcv/datasets/C5/KITTI-MOTS"

def get_instance_images(instance_id: int):
    masks_filenames = list(filter(lambda x: x.endswith(".png"), sorted(os.listdir(os.path.join(dataset_path, f"instances/{int(instance_id):04d}")))))
    images_filenames = list(filter(lambda x: x.endswith(".png"), sorted(os.listdir(os.path.join(dataset_path, f"training/image_02/{int(instance_id):04d}")))))

    
    masks = [Image.open(os.path.join(dataset_path, f"instances/{int(instance_id):04d}", filename)) for filename in masks_filenames]
    images = [Image.open(os.path.join(dataset_path, f"training/image_02/{int(instance_id):04d}", filename)) for filename in images_filenames]
   
    return images, masks

train_images, train_annotations = [], []
test_images, test_annotations = [], []

for instance in train_instances_ids:
    i, m = get_instance_images(instance)
    train_images.extend(i)
    train_annotations.extend(m)

for instance in test_instances_ids:
    i, m = get_instance_images(instance)
    test_images.extend(i)
    test_annotations.extend(m)

train_split = {
    "image": train_images,
    "annotation": train_annotations,
}

validation_split = {
    "image": test_images,
    "annotation": test_annotations,
}

def create_instance_segmentation_dataset(label2id, **splits):
    dataset_dict = {}
    for split_name, split in splits.items():
        split["semantic_class_to_id"] = [label2id] * len(split["image"])
        dataset_split = (
            Dataset.from_dict(split)
            .cast_column("image", DatasetImage())
            .cast_column("annotation", DatasetImage())
        )
        dataset_dict[split_name] = dataset_split
    return DatasetDict(dataset_dict)

dataset = create_instance_segmentation_dataset(label2id, train=train_split, validation=validation_split)
dataset.push_to_hub("marcagve18/kitti-mots-instance-seg")