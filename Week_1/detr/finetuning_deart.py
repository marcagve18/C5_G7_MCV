import torch
from transformers import AutoImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer
from data_processing import KITTIMOTS_CocoDetection
import evaluate
from tqdm import tqdm
import numpy as np
import albumentations
from datasets import load_dataset

transform = albumentations.Compose(
    [
        albumentations.HorizontalFlip(p=1.0),
        albumentations.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
)


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch

def transform_aug_ann(examples):
    images = [np.array(image.convert("RGB"))[:, :, ::-1] for image in examples['image']]
    annotations = []

    for img_id, img_annotations in zip(examples['image_id'], examples['annotations']):
        if len(img_annotations) > 0:
            annotations.append({
                "image_id": img_id,
                "annotations": img_annotations
            })
        else:
            annotations.append({
                "image_id": img_id,
                "annotations": []
            })
    return image_processor(images=images, annotations=annotations, return_tensors="pt")


# Set up the device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image processor and model, then move model to the device
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

model.to(device)

# Define dataset path and load your preprocessed Hugging Face dataset
dataset_path = "/home/mcv/datasets/C5/KITTI-MOTS"
test_instances_ids = [10, 6, 2, 16, 0, 17, 3, 14, 12]
test_dataset = KITTIMOTS_CocoDetection(dataset_path, instances_ids=test_instances_ids)
deart_dataset = load_dataset('davanstrien/deart')['train'].with_transform(transform_aug_ann)

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
    train_dataset=deart_dataset,
    eval_dataset=test_dataset,
    processing_class=image_processor,
)

trainer.train()