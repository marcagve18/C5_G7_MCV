import torch
from transformers import AutoImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer
from data_processing import KITTIMOTS_CocoDetection
import evaluate
from tqdm import tqdm
import numpy as np
import albumentations

transform = albumentations.Compose(
    [
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)

def merge_labels(labels_list):
    """
    Merge a list of annotation dictionaries into a single dictionary.
    Assumes all dictionaries have the same keys.
    """
    merged = {}
    for key in labels_list[0].keys():
        merged[key] = torch.cat([d[key] for d in labels_list], dim=0)
    return merged

def collate_fn(batch):
    pixel_values = []
    for item in batch:
        img = item["pixel_values"]
        # Ensure the image is (C, H, W)
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        pixel_values.append(img)
        
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    
    processed_labels = []
    for item in batch:
        label = item["labels"]
        if isinstance(label, dict):
            processed_labels.append(label)
        elif isinstance(label, list):
            processed_labels.append(merge_labels(label))
        else:
            raise ValueError("Unexpected label format")
    
    output = {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": processed_labels
    }
    return output


# Set up the device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image processor and model, then move model to the device
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

model.to(device)

# Define dataset path and load your preprocessed Hugging Face dataset
dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
train_instances_ids = [19, 20, 9, 7, 1, 8, 15, 11, 13, 18, 4, 5]
test_instances_ids = [10, 6, 2, 16, 0, 17, 3, 14, 12]


train_dataset = KITTIMOTS_CocoDetection(dataset_path, instances_ids=train_instances_ids, transforms=transform)
test_dataset = KITTIMOTS_CocoDetection(dataset_path, instances_ids=test_instances_ids)
print("HF Datasets loaded")

training_args = TrainingArguments(
    output_dir="checkpoints/first-finetune-detr",
    per_device_train_batch_size=8,
    num_train_epochs=1,
    fp16=True,
    save_steps=200,
    logging_steps=10,           # More frequent logging
    logging_first_step=True,    # Log the very first step
    logging_dir="logs",
    learning_rate=1e-5,
    weight_decay=1e-4,
    save_total_limit=2,
    remove_unused_columns=False,
    report_to=[],               # Disable external logging integrations
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=image_processor,
)

trainer.train()