import torch
from datetime import datetime
import sys
from Image_Captioning_Utils.utils import get_train_val_test_annotations_split
from Image_Captioning_Utils.dataset import FoodDatasetWordLevel
from torch.utils.data import DataLoader
from train_model import initialize_models, create_data_loaders, validate_model, training_config  

# Add project path
sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load checkpoint ===
checkpoint_path = "/ghome/c5mcv07/C5_G7_MCV/Week_4/LLMs/checkpoints/2025-04-06_19-49-12_meta-llama_Llama-3.2-3B_best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location=device)

# === Initialize model architecture ===
llama_model_name = "meta-llama/Llama-3.2-3B"
processor, vit_model, peft_model, projection_layer, embedding_layer = initialize_models(llama_model_name)

# === Load checkpoint weights ===
peft_model.load_state_dict(checkpoint['model_state_dict'])
projection_layer.load_state_dict(checkpoint['projection_state_dict'])

# Set to eval mode
peft_model.eval()
projection_layer.eval()
vit_model.eval()

# === Load test data ===
splits = get_train_val_test_annotations_split()
test_dataset = FoodDatasetWordLevel(splits["test"])

def custom_collate_fn(batch):
    images, cap_idx, captions = zip(*batch)
    pixel_values = processor(images=list(images), return_tensors="pt").pixel_values.squeeze()
    cap_idxs = torch.stack(cap_idx)
    return {
        "pixel_values": pixel_values,
        "captions": captions,
        "cap_idxs": cap_idxs,
    }

test_loader = DataLoader(
    test_dataset,
    batch_size=4,
    shuffle=False,
    collate_fn=custom_collate_fn,
    num_workers=1
)

# === Evaluate ===
with torch.no_grad():
    metrics = validate_model(
        model=peft_model,
        val_loader=test_loader,
        vit_model=vit_model,
        projection_layer=projection_layer,
        embedding_layer=embedding_layer
    )

# === Print results ===
print("Evaluation metrics on test set:")
for key, value in metrics.items():
    print(f"{key}: {value}")
