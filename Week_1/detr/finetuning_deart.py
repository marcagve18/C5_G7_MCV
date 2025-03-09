import torch
from transformers import AutoImageProcessor, DetrForObjectDetection, TrainingArguments, Trainer
from data_processing import KITTIMOTS_CocoDetection
from tqdm import tqdm
import numpy as np
import albumentations
from datasets import load_dataset
from dotenv import load_dotenv
import os
import wandb
from transformers import set_seed

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

EXPERIMENT_NAME = "DEART_v1"

WANDB_KEY = os.getenv('WANDB_MARC')
wandb.login(key=WANDB_KEY)

import albumentations as A

train_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=1200),
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
)

test_transform = A.Compose(
    [
        A.LongestMaxSize(max_size=1200),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category_ids"]),
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
    images_aug = []
    annotations_aug = []
    
    for img, img_id, anns in zip(examples['image'], examples['image_id'], examples['annotations']):
        # Convert image to numpy array in BGR format (Albumentations uses BGR by default).
        image_np = np.array(img.convert("RGB"))[:, :, ::-1]
        
        if len(anns) > 0:
            # Extract bounding boxes and labels.
            bboxes = []
            category_ids = []
            for ann in anns:
                bboxes.append(ann["bbox"])
                category_ids.append(ann["category_id"])
            
            augmented = train_transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
            image_transformed = augmented["image"]
            new_bboxes = augmented["bboxes"]

            new_anns = anns
            for i, ann in enumerate(anns):
                new_anns[i]['bbox'] = new_bboxes[i]
        else:
            # No annotations: just apply image augmentation.
            image_transformed = image_np
            new_anns = []
        
        images_aug.append(image_transformed)
        annotations_aug.append({
            "image_id": img_id,
            "annotations": new_anns
        })
    
    return image_processor(images=images_aug, annotations=annotations_aug, return_tensors="pt")



def test_transform_ann(examples):
    images_aug = []
    annotations_aug = []
    
    for img, img_id, anns in zip(examples['image'], examples['image_id'], examples['annotations']):
        # Convert image to numpy array in BGR format (Albumentations uses BGR by default).
        image_np = np.array(img.convert("RGB"))[:, :, ::-1]
        
        if len(anns) > 0:
            # Extract bounding boxes and labels.
            bboxes = []
            category_ids = []
            for ann in anns:
                bboxes.append(ann["bbox"])
                category_ids.append(ann["category_id"])
            
            augmented = test_transform(image=image_np, bboxes=bboxes, category_ids=category_ids)
            image_transformed = augmented["image"]
            new_bboxes = augmented["bboxes"]

            new_anns = anns
            for i, ann in enumerate(anns):
                new_anns[i]['bbox'] = new_bboxes[i]
        else:
            # No annotations: just apply image augmentation.
            image_transformed = image_np
            new_anns = []
        
        images_aug.append(image_transformed)
        annotations_aug.append({
            "image_id": img_id,
            "annotations": new_anns
        })
    
    return image_processor(images=images_aug, annotations=annotations_aug, return_tensors="pt")



# Set up the device: GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load image processor and model, then move model to the device
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

model.to(device)

# Define dataset path and load your preprocessed Hugging Face dataset
dataset = load_dataset("davanstrien/deart")
split_dataset = dataset["train"].train_test_split(test_size=0.05, seed=42)
train_dataset = split_dataset['train'].with_transform(transform_aug_ann)
test_dataset = split_dataset['test'].with_transform(test_transform_ann)

print("HF Datasets loaded")

training_config = {
    "output_dir": f"checkpoints/{EXPERIMENT_NAME}",
    "per_device_train_batch_size": 8,
    #"gradient_accumulation_steps": 2,
    "num_train_epochs": 20,
    "fp16": True,
    "save_steps": 100,
    "logging_steps": 1,
    "evaluation_strategy": "steps",  # Evaluate during training
    "eval_steps": 50,
    "logging_first_step": True,    # Log the very first step
    "learning_rate": 5e-6,
    "weight_decay": 1e-4,
    "save_total_limit": 10,
    "remove_unused_columns": False,
    "report_to": ['wandb'],
    "load_best_model_at_end": True,      
    "metric_for_best_model": "eval_loss",
    "greater_is_better": False, 
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