import argparse
import os
import random
import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# Add custom module path if needed
import sys
sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")
from Image_Captioning_Utils.dataset import FoodDatasetWord
from Image_Captioning_Utils.utils import get_train_val_test_annotations_split

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def main():
    parser = argparse.ArgumentParser(description="Inference Script for Sampling 50 Random Test Images")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the model checkpoint directory")
    parser.add_argument("--output_dir", type=str, default="inference_results",
                        help="Directory to save the output figures")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of random test images to sample")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model, feature extractor, and tokenizer
    model = VisionEncoderDecoderModel.from_pretrained(args.checkpoint)
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Generation parameters (adjust as needed)
    gen_kwargs = {
        "max_length": 16, # Allow slightly longer for sampling
        "num_beams": 1, # MUST be 1 for sampling
        "do_sample": True,
        "top_k": 50,
        "top_p": 0.9, # Slightly tighter p
        "temperature": 0.2, # Reduce temperature for less randomness
        "repetition_penalty": 1.3, # Reduce penalty
        "no_repeat_ngram_size": 2, # Try 2 first
    }

    gen_kwargs = {"max_length": 16,
    "num_beams": 4,
    "do_sample": False,
    "top_k": 50,    # or an appropriate value
    "top_p": 0.95,
    "repetition_penalty": 1.5,
    "no_repeat_ngram_size": 2}

    # Load test split annotations and create the test dataset
    splits = get_train_val_test_annotations_split()
    test_annotations = splits["test"]
    test_dataset = FoodDatasetWord(test_annotations)

    # Sample num_samples random indices from the test dataset
    total_samples = len(test_dataset)
    if args.num_samples > total_samples:
        raise ValueError(f"Requested {args.num_samples} samples, but test dataset only has {total_samples} images.")
    sampled_indices = random.sample(range(total_samples), args.num_samples)

    # Loop over sampled indices, run inference, and save figures
    for idx, sample_idx in enumerate(sampled_indices):
        # Retrieve the sample: assume each sample is a tuple (PIL.Image, caption)
        image, true_caption = test_dataset[sample_idx]

        # Process the image using the feature extractor (expects a list of images)
        pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values.to(device)

        # Run inference with no_grad for efficiency
        with torch.no_grad():
            output_ids = model.generate(pixel_values, **gen_kwargs)
        pred_caption = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Create and save a figure for the current sample
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.title(f"Predicted: {pred_caption}\nGround Truth: {true_caption}", fontsize=10)
        plt.axis("off")
        output_path = os.path.join(args.output_dir, f"sample_{idx+1:02d}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()  # Close the figure to free memory
        print(f"Saved sample {idx+1} to {output_path}")

if __name__ == "__main__":
    main()
