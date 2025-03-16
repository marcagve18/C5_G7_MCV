#!/usr/bin/env python

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from tqdm import tqdm

# Create output directory for comparisons
os.makedirs("outputs_comparison", exist_ok=True)

# Set device: use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def map_label(label):
    """
    Map labels so that 'pedestrian' becomes 'person'.
    """
    if label.lower() == "pedestrian":
        return "person"
    return label

def get_segmentation_results(image, processor, model, threshold=0.9):
    """
    Process an image and return the segmentation results.
    Moves input tensors to the GPU if available.
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]], threshold=threshold
    )[0]
    return results

def compute_total_confidence(results, model, allowed_classes):
    """
    Compute the total confidence (sum of scores) for allowed classes
    in a single image.
    """
    total = 0
    for seg in results["segments_info"]:
        label = model.config.id2label[seg["label_id"]]
        mapped_label = map_label(label).lower()
        if mapped_label in allowed_classes:
            total += seg.get("score", 0)
    return total

def plot_side_by_side_overlay(image, baseline_results, finetuned_results,
                              baseline_model, finetuned_model, output_path):
    """
    Plots a side-by-side overlay of segmentation results.
    Left: baseline; Right: finetuned.
    """
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    def plot_overlay(ax, image, results, model, title):
        img_np = np.array(image)
        ax.imshow(img_np)
        ax.axis("off")
        # Map segment id to label id
        segment_to_label = {seg["id"]: seg["label_id"] for seg in results["segments_info"]}
        for seg in results["segments_info"]:
            seg_id = seg["id"]
            label = model.config.id2label[segment_to_label[seg_id]]
            label = map_label(label)
            color = [random.random() for _ in range(3)]
            # Get mask as numpy array
            mask = (results["segmentation"].cpu().numpy() 
                    if isinstance(results["segmentation"], torch.Tensor)
                    else results["segmentation"]) == seg_id
            colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
            colored_mask[mask, :3] = color
            colored_mask[mask, 3] = 0.9
            ax.imshow(colored_mask, interpolation="none")
            ys, xs = np.where(mask)
            if len(ys) > 0 and len(xs) > 0:
                ax.text(xs.mean(), ys.mean(), f"#{seg_id}:{label}",
                        color="white", fontsize=12, ha="center", va="center",
                        bbox=dict(facecolor="black", alpha=0.5, pad=1))
        ax.set_title(title)
    
    plot_overlay(axs[0], image, baseline_results, baseline_model, "Baseline")
    plot_overlay(axs[1], image, finetuned_results, finetuned_model, "Finetuned")
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def filter_allowed(results, model, allowed):
    """
    Filters segmentation results to keep only segments where the label (after mapping)
    is in allowed classes.
    """
    filtered = []
    for s in results["segments_info"]:
        label = model.config.id2label[s["label_id"]].lower()
        if label in allowed or (label == "pedestrian" and "person" in allowed):
            filtered.append(s)
    results["segments_info"] = filtered
    return results

def main():
    # Allowed classes: merging "pedestrian" into "person"
    allowed_classes = ["car", "person"]

    # Load evaluation image paths (adjust directory as needed)
    eval_img_paths = []
    instances_path = "/ghome/c5mcv07/mcv/datasets/C5/KITTI-MOTS/testing/image_02"
    for instance_id in os.listdir(instances_path):
        instance_folder = os.path.join(instances_path, instance_id)
        for file in os.listdir(instance_folder):
            if file.endswith(".png"):
                eval_img_paths.append(os.path.join(instance_folder, file))
    eval_img_paths = eval_img_paths[:4000]  # Limit to 100 images for evaluation

    # Load baseline model and its processor
    baseline_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
    baseline_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-tiny-coco-instance")
    baseline_model.to(device)

    # Load finetuned model and its processor (adjust the paths accordingly)
    finetuned_processor = AutoImageProcessor.from_pretrained("/ghome/c5mcv07/C5_G7_MCV/Week_2/huggingface/mask2former/checkpoints/Mask2Former_KITTI_v1/checkpoint-2000")
    finetuned_model = Mask2FormerForUniversalSegmentation.from_pretrained("/ghome/c5mcv07/C5_G7_MCV/Week_2/huggingface/mask2former/checkpoints/Mask2Former_KITTI_v1/checkpoint-2000")
    finetuned_model.to(device)

    # Collect difference examples as tuples:
    # (difference, img_path, baseline_results, finetuned_results, image)
    difference_examples = []

    for idx, img_path in enumerate(tqdm(eval_img_paths)):
        with Image.open(img_path) as image:
            image_copy = image.copy()
        # Obtain segmentation results from both models
        baseline_results = get_segmentation_results(image_copy, baseline_processor, baseline_model, threshold=0.9)
        finetuned_results = get_segmentation_results(image_copy, finetuned_processor, finetuned_model, threshold=0.9)
        # Filter for allowed classes
        baseline_results = filter_allowed(baseline_results, baseline_model, allowed_classes)
        finetuned_results = filter_allowed(finetuned_results, finetuned_model, allowed_classes)
        # Compute total confidence for allowed classes
        baseline_conf = compute_total_confidence(baseline_results, baseline_model, allowed_classes)
        finetuned_conf = compute_total_confidence(finetuned_results, finetuned_model, allowed_classes)
        diff = abs(baseline_conf - finetuned_conf)
        difference_examples.append((diff, img_path, baseline_results, finetuned_results, image_copy))

    # Sort by difference (largest differences first)
    difference_examples.sort(key=lambda x: x[0], reverse=True)

    # Choose top N examples to display
    top_n = 50
    top_examples = difference_examples[:top_n]

    for i, (diff, img_path, baseline_res, finetuned_res, image) in enumerate(top_examples):
        out_path = os.path.join("outputs_comparison", f"side_by_side_example_{i}.png")
        plot_side_by_side_overlay(image, baseline_res, finetuned_res, baseline_model, finetuned_model, out_path)
        print(f"Saved side-by-side comparison for image {img_path} with difference {diff:.3f} to {out_path}")

if __name__ == "__main__":
    main()
