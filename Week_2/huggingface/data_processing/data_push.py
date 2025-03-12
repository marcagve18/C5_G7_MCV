from datasets import Dataset, DatasetDict
from datasets import Image as DatasetImage
from PIL import Image
import os
from huggingface_hub import login


HF_TOKEN = ''
login(HF_TOKEN)

train_instances_ids = [19, 20, 9, 7, 1, 8, 15, 11, 13, 18, 4, 5]
test_instances_ids = [10, 6, 2, 16, 0, 17, 3, 14, 12]

label2id = {
    "background": 0,
    "pedestrian": 2,
    "car": 1,
}

dataset_path = "/ghome/c5mcv07/C5_G7_MCV/Week_2/huggingface/processed_datasets/KITTI_MOTS"

def get_instance_images(instance_id: int):
    files = sorted(os.listdir(os.path.join(dataset_path, f"{int(instance_id):04d}")))

    masks_filenames = list(filter(lambda x: x.endswith("_mask.png"), files))
    images_filenames = list(filter(lambda x: not x.endswith("_mask.png"), files))
    
    masks = [Image.open(os.path.join(dataset_path, f"{int(instance_id):04d}", filename)) for filename in masks_filenames]
    images = [Image.open(os.path.join(dataset_path, f"{int(instance_id):04d}", filename)) for filename in images_filenames]
   
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