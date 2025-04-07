import torch
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import numpy as np

sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")

from Image_Captioning_Utils.metrics import calculate_metrics
from Image_Captioning_Utils.dataset import FoodDatasetWord
from Image_Captioning_Utils.utils import get_train_val_test_annotations_split


SEED = 42

# Set seeds for Python, NumPy, and PyTorch
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and processor
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"  # https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)

def custom_collate_fn(batch):
    images, captions = zip(*batch)

    return {
        "images": images,
        "captions": captions,
    }

def predict_step(images):
    # Create messages for each image in the batch
    messages = [
        [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Give me a short title or name for the recipe shown in this image. Do not use double quotes."}
            ]
        }] for img in images
    ]

    # Apply the chat template to generate text prompts for each image
    text_prompts = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process the image inputs and handle any video inputs (though it's not used here)
    image_inputs, video_inputs = process_vision_info(messages)

    # Prepare the inputs for the model, passing both the text and image data
    inputs = processor(
        text=text_prompts,  # List of text prompts for each image
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)

    # Generate output from the model
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    # Trim the generated IDs to remove the input tokens and keep only the output
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    # Decode the generated token IDs back into readable text (captions)
    predictions = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Return the captions (one per image in the batch)
    return [pred.strip() for pred in predictions]

# Load dataset
splits = get_train_val_test_annotations_split()
test_annotations = splits["test"]
test_dataset = FoodDatasetWord(test_annotations)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=custom_collate_fn)

# Evaluate on the test set
all_true_captions = []
all_predicted_captions = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader):
        images = batch["images"]  # PIL images
        true_captions = batch["captions"]

        preds = predict_step(images)
        all_true_captions.extend(true_captions)
        all_predicted_captions.extend(preds)

for true_caption, predicted_caption in zip(all_true_captions, all_predicted_captions):
    print(f"true_caption: {true_caption} -> predicted_caption: {predicted_caption}")

# Calculate and print evaluation metrics
metrics = calculate_metrics(all_true_captions, all_predicted_captions)
print("Evaluation Metrics:")
for metric, score in metrics.items():
    print(f"{metric}: {score}")
