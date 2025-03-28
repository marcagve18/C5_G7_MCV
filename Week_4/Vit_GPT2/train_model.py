import wandb  
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")
from Image_Captioning_Utils.dataset import FoodDatasetWord
from Image_Captioning_Utils.utils import get_train_val_test_annotations_split
from Image_Captioning_Utils.constants import TEXT_MAX_LEN
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Freeze the decoder parameters so that only the ViT encoder is trained
for param in model.decoder.parameters():
    param.requires_grad = False

from Image_Captioning_Utils.constants import TEXT_MAX_LEN

def custom_collate_fn(batch):
    # Each element in the batch is assumed to be a tuple: (image, caption)
    images, captions = zip(*batch)
    
    # Process images into pixel values using the ViT feature extractor
    pixel_values = feature_extractor(images=list(images), return_tensors="pt").pixel_values

    # Tokenize the captions using the pre-trained tokenizer
    tokenized = tokenizer(
        list(captions),
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LEN,
        return_tensors="pt"
    )

    # Use the tokenized input IDs as labels
    cap_idxs = tokenized["input_ids"]

    return {
        "pixel_values": pixel_values,
        "captions": captions,
        "cap_idxs": cap_idxs,
    }

# Get train and validation splits (adjust if you want a validation loop)
splits = get_train_val_test_annotations_split()
train_annotations = splits["train"]
val_annotations = splits["val"]

training_config = {
    "lr": 1e-4,
    "batch_size": 40,
    "epochs": 10,
}

num_workers = 8  # or os.cpu_count()

train_dataset = FoodDatasetWord(train_annotations)
val_dataset = FoodDatasetWord(val_annotations)

train_loader = DataLoader(train_dataset, batch_size=training_config['batch_size'], shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=training_config['batch_size'], shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)

# Define optimizer for the encoder parameters only
optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=training_config['lr'])
num_epochs = training_config['epochs']

# Initialize wandb run
wandb.init(
    name="ViT_Train_1",
    project="C5_W4_ViT-GPT2", 
    config=training_config
)
# Optional: Watch the model to log gradients and parameters
wandb.watch(model, log="all", log_freq=100)

best_val_loss = float('inf')
global_step = 0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        optimizer.zero_grad()
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["cap_idxs"].to(device)
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        global_step += 1
        # Log every 5 steps
        if global_step % 5 == 0:
            wandb.log({"step": global_step, "train_loss": loss.item(), "epoch": (global_step) / (len(train_loader))})
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
    wandb.log({"epoch": epoch+1, "avg_train_loss": avg_train_loss})
    
    model.eval()
    val_running_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["cap_idxs"].to(device)
            labels[labels == tokenizer.pad_token_id] = -100
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            val_running_loss += loss.item()
            
    avg_val_loss = val_running_loss / len(val_loader)
    print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
    wandb.log({"epoch": epoch+1, "val_loss": avg_val_loss})

    # Check if the current model has the best validation loss so far, and save if it does.
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), f'checkpoints/vit/model_epoch_{epoch+1}.pth')
        print(f"Saved best model at epoch {epoch+1} with val_loss: {avg_val_loss:.4f}")