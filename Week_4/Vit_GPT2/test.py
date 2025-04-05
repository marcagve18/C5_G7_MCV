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

model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the checkpoint
checkpoint_path = None
if checkpoint_path is not None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)

    # Optionally, set to evaluation mode
    model.eval()

    print("Checkpoint loaded successfully!")

max_length = 32
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

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

# Evaluate on the test set
all_true_captions = []
all_predicted_captions = []

model.eval()
with torch.no_grad():
    for batch_idx, batch in enumerate(tqdm(test_loader)):
        # Adjust these keys if your dataset returns a tuple
        image_pixels = batch["pixel_values"].to(device)  # list of image paths
        true_captions = batch["captions"]     # list of corresponding ground truth captions
        
        preds = predict_step(image_pixels)
        all_true_captions.extend(true_captions)
        all_predicted_captions.extend(preds)



# Calculate and print evaluation metrics
metrics = calculate_metrics(all_true_captions, all_predicted_captions)
print("Evaluation Metrics:")
for metric, score in metrics.items():
    print(f"{metric}: {score}")
