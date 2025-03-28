from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Add your module path (if needed)
sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")

from Image_Captioning_Utils.dataset import FoodDatasetWord
from Image_Captioning_Utils.utils import get_train_val_test_annotations_split
from Image_Captioning_Utils.constants import TEXT_MAX_LEN

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

num_workers = 8 #os.cpu_count()

train_dataset = FoodDatasetWord(train_annotations)
val_dataset = FoodDatasetWord(val_annotations)

train_loader = DataLoader(train_dataset, batch_size=40, shuffle=True, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False, collate_fn=custom_collate_fn, num_workers=num_workers, pin_memory=True, persistent_workers=True)

# Define optimizer for the encoder parameters only
optimizer = torch.optim.AdamW(model.encoder.parameters(), lr=1e-4)
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        optimizer.zero_grad()
        # Move pixel values to the appropriate device
        pixel_values = batch["pixel_values"].to(device)
        # Get the tokenized captions and move them to device
        labels = batch["cap_idxs"].to(device)
        
        # Replace pad token ids with -100 so that they are ignored in the loss computation.
        labels[labels == tokenizer.pad_token_id] = -100

        # Forward pass: passing labels makes the model compute the loss internally.
        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")