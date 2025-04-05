import wandb  
from transformers import (
    VisionEncoderDecoderModel, 
    ViTImageProcessor, 
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
from torchvision import transforms

sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")
from Image_Captioning_Utils.dataset import FoodDatasetWord
from Image_Captioning_Utils.utils import get_train_val_test_annotations_split
import numpy as np
from dotenv import load_dotenv

load_dotenv()
SEED = 42

# Set seeds for Python, NumPy, and PyTorch
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

WANDB_KEY = os.getenv('WANDB_MARC')
wandb.login(key=WANDB_KEY)

# Load model components
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    print(f"Set tokenizer pad_token to eos_token: {tokenizer.pad_token}")

# Update model config for generation
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

TEXT_MAX_LEN = 16

augmentation_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),   # Adjust size as needed.
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
])

def custom_collate_fn(batch):
    # Each element in the batch is assumed to be a tuple: (image, caption)
    images, captions = zip(*batch)

    augmented_images = [augmentation_transforms(img) for img in images]
    
    # Process images into pixel values using the ViT feature extractor
    pixel_values = feature_extractor(images=augmented_images, return_tensors="pt").pixel_values

    # Tokenize the captions using the pre-trained tokenizer
    tokenized = tokenizer(
        list(captions),
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LEN,
        return_tensors="pt",
    )
    # Return tokenized input_ids as "labels"
    return {
        "pixel_values": pixel_values,
        "labels": tokenized["input_ids"],
        #"captions": captions  # optional, for debugging
    }

# Get train and validation splits
splits = get_train_val_test_annotations_split()
train_annotations = splits["train"]
val_annotations = splits["val"]

training_config = {
    "lr": 1e-5,
    "batch_size": 40,
    "epochs": 20,
    "weight_decay": 1e-5
}

num_workers = 8  # or os.cpu_count()

train_dataset = FoodDatasetWord(train_annotations)
val_dataset = FoodDatasetWord(val_annotations)

# (We no longer create explicit DataLoaders since Seq2SeqTrainer will handle it via our custom collate_fn)

part_to_train = "all"  # choose from "encoder", "decoder", or "all"

# Freeze parameters accordingly
if part_to_train == "encoder":
    for param in model.decoder.parameters():
        param.requires_grad = False
elif part_to_train == "decoder":
    for param in model.encoder.parameters():
        param.requires_grad = False
elif part_to_train == "all":
    pass
else:
    raise ValueError("Part to train must be either: encoder, decoder or all")

print("TRAINING", part_to_train)
num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {num_trainable_params}")

experiment_name = f"{part_to_train}_long_data_aug"

wandb.init(
    name=experiment_name,
    project="C5_W4_ViT-GPT2", 
    config=training_config
)

wandb.watch(model, log="all", log_freq=100)

gen_kwargs = {"max_length": TEXT_MAX_LEN, "num_beams": 4}

# Define custom compute_metrics function using your calculate_metrics
def compute_metrics_func(eval_pred):
    preds = eval_pred.predictions
    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    labels = eval_pred.label_ids
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    # Decode the generated predictions and the ground truth labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    from Image_Captioning_Utils.metrics import calculate_metrics
    metrics = calculate_metrics(decoded_labels, decoded_preds)
    # You can log additional info if needed
    for metric, score in metrics.items():
        print(f"{metric}: {score}")
    return metrics

# Define Seq2SeqTrainingArguments (adjust eval_steps and logging_steps as desired)
training_args = Seq2SeqTrainingArguments(
    output_dir=f"checkpoints/{part_to_train}/{experiment_name}",
    learning_rate=training_config['lr'],
    num_train_epochs=training_config['epochs'],
    per_device_train_batch_size=training_config['batch_size'],
    per_device_eval_batch_size=training_config['batch_size'],
    weight_decay=training_config['weight_decay'],
    eval_strategy="steps",
    eval_steps=100,            # evaluate every 100 steps
    logging_steps=1,         # log every 100 steps
    save_steps=100,            # save every 100 steps
    predict_with_generate=True,
    logging_dir="./logs",
    save_total_limit=3,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="meteor",  
    greater_is_better=True        
)

# Instantiate the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_collate_fn,
    compute_metrics=compute_metrics_func,
)

# Start training using the Trainer API
trainer.train()

trainer.save_model(f"checkpoints/{part_to_train}/{experiment_name}")
