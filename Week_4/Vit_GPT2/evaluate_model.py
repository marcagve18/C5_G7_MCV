from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm

sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")

from Image_Captioning_Utils.metrics import calculate_metrics
from Image_Captioning_Utils.dataset import FoodDatasetWord
from Image_Captioning_Utils.utils import get_train_val_test_annotations_split
import numpy as np

SEED = 42

# Set seeds for Python, NumPy, and PyTorch
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

#model_name = "nlpconnect/vit-gpt2-image-captioning"
model_name = "/ghome/c5mcv07/C5_G7_MCV/Week_4/Vit_GPT2/checkpoints/decoder/decoder_long_saved"
#model_name = "/ghome/c5mcv07/C5_G7_MCV/Week_4/Vit_GPT2/checkpoints/vit-gpt2-food-captioning-two-stage/stage2_full_finetune/best_model_final"
#model_name = "/ghome/c5mcv07/C5_G7_MCV/Week_4/Vit_GPT2/checkpoints/encoder/deleteme_encoder/checkpoint-2500"
#model_name = "/ghome/c5mcv07/C5_G7_MCV/Week_4/Vit_GPT2/checkpoints/all/deleteme_all/checkpoint-2700"
#model_name = "/ghome/c5mcv07/C5_G7_MCV/Week_4/Vit_GPT2/checkpoints/decoder/decoder_long_data_aug"
#model_name = "/ghome/c5mcv07/C5_G7_MCV/Week_4/Vit_GPT2/checkpoints/all/all_long_data_aug"
#model_name = "/ghome/c5mcv07/C5_G7_MCV/Week_4/Vit_GPT2/checkpoints/all/all_long_data_fix"
#model_name = "/ghome/c5mcv07/C5_G7_MCV/Week_4/Vit_GPT2/checkpoints/all/all_long_data_fix/checkpoint-600"
#model_name = "/ghome/c5mcv07/C5_G7_MCV/Week_4/Vit_GPT2/checkpoints/decoder/decoder_large"

model = VisionEncoderDecoderModel.from_pretrained(model_name)
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


model.eval()
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length,
    "num_beams": num_beams,
    "do_sample": False,
}

'''gen_kwargs = {
    "max_length": max_length, # Allow slightly longer for sampling
    "num_beams": 1, # MUST be 1 for sampling
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.9, # Slightly tighter p
    "temperature": 0.7, # Reduce temperature for less randomness
    "repetition_penalty": 1.3, # Reduce penalty
    "no_repeat_ngram_size": 1, # Try 2 first
}'''

def custom_collate_fn(batch):
    # Assume each element in batch is a tuple: (image, caption)
    images, captions = zip(*batch)
    
    # Use the ViT preprocessor to convert a list of PIL images to pixel values
    pixel_values = feature_extractor(images=list(images), return_tensors="pt").pixel_values
        
    # Return a dictionary with the processed pixel values and the other data
    return {
        "pixel_values": pixel_values,
        "captions": captions,
    }

def predict_step(pixel_values):
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

splits = get_train_val_test_annotations_split()
train_annotations = splits["train"]
val_annotations = splits["val"]
test_annotations = splits["test"]

# Create dataset and dataloader
train_dataset = FoodDatasetWord(train_annotations)
val_dataset = FoodDatasetWord(val_annotations)
test_dataset = FoodDatasetWord(test_annotations)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)

def evaluate_model(model, dataloader, device):
    all_true_captions = []
    all_predicted_captions = []

    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            image_pixels = batch["pixel_values"].to(device)
            true_captions = batch["captions"]

            preds = predict_step(image_pixels)
            all_true_captions.extend(true_captions)
            all_predicted_captions.extend(preds)

    metrics = calculate_metrics(all_predicted_captions, all_true_captions)
    return metrics, all_true_captions, all_predicted_captions

# Train split
'''train_metrics, train_true, train_pred = evaluate_model(model, train_loader, device)
print("\nTrain Evaluation Metrics:")
for metric, score in train_metrics.items():
    print(f"{metric}: {score}")'''

# Validation split
val_metrics, val_true, val_pred = evaluate_model(model, val_loader, device)
print("\nValidation Evaluation Metrics:")
for metric, score in val_metrics.items():
    print(f"{metric}: {score}")

# Test split
test_metrics, test_true, test_pred = evaluate_model(model, test_loader, device)
print("\nTest Evaluation Metrics:")
for metric, score in test_metrics.items():
    print(f"{metric}: {score}")