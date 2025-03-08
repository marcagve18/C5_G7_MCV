import torch
from transformers import AutoImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer, set_seed
from data_processing import KITTIMOTS_CocoDetection
import albumentations
from dotenv import load_dotenv
import os
import wandb
from transformers import set_seed
import numpy as np

SEED = 42

# Set seeds for Python, NumPy, and PyTorch
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

# Set seed for Hugging Face transformers (affects some internal RNGs)
set_seed(SEED)

load_dotenv()

EXPERIMENT_NAME = "KITTI_v2_lower_lr"

WANDB_KEY = os.getenv('WANDB_MARC')
wandb.login(key=WANDB_KEY)

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

training_config = {
    "output_dir": f"checkpoints/{EXPERIMENT_NAME}",
    "per_device_train_batch_size": 8,
    "num_train_epochs": 20,
    "fp16": True,
    "save_steps": 100,
    "logging_steps": 1,
    "evaluation_strategy": "steps",  # Evaluate during training
    "eval_steps": 30,
    "logging_first_step": True,    # Log the very first step
    "learning_rate": 5e-6,
    "weight_decay": 1e-4,
    "save_total_limit": 2,
    "remove_unused_columns": False,
    "report_to": ['wandb'],
}

wandb.init(name=EXPERIMENT_NAME, project="C5_W1_DETR", config=training_config)

training_args = TrainingArguments(**training_config)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=image_processor,
)

trainer.train()