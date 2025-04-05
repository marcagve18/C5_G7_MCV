from base64 import decode
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

sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")
from Image_Captioning_Utils.dataset import FoodDatasetWord
from Image_Captioning_Utils.utils import get_train_val_test_annotations_split
import numpy as np
from dotenv import load_dotenv

# Load model components
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False

def get_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

freeze_model(model)
initial_params = get_trainable_params(model)

for param in model.decoder.parameters():
    param.requires_grad = True

decoder_params = get_trainable_params(model)
freeze_model(model)

for param in model.encoder.parameters():
    param.requires_grad = True

encoder_params = get_trainable_params(model)

for param in model.parameters():
    param.requires_grad = True

total_params = get_trainable_params(model)

assert encoder_params + decoder_params == total_params

print(f" - Encoder params: {encoder_params} ({(encoder_params/total_params)*100}%)")
print(f" - Decoder params: {decoder_params} ({(decoder_params/total_params)*100}%)")

