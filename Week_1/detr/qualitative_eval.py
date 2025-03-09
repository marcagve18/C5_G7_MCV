import os
import glob
import torch
from transformers import AutoImageProcessor, DetrForObjectDetection
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from tqdm import tqdm  # for progress bar

# ----------------------------
# Configuration
# ----------------------------
# Directory pattern for evaluation images
eval_pattern = "/ghome/c5mcv07/mcv/datasets/C5/KITTI-MOTS/testing/image_02/*/*.png"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Filter criteria: only "car" and "person"
desired_categories = {"car", "person"}

# Number of images to save with lowest and highest average confidence
num_examples = 20

# ----------------------------
# Setup model and image processor
# ----------------------------
checkpoint = 'checkpoints/KITTI_MOTS_v1/checkpoint-3500'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained(checkpoint).to(device)

# ----------------------------
# Prepare lists to aggregate detection data
# ----------------------------
all_detection_scores = []
all_detection_labels = []
all_detection_areas = []

# List to store per-image info for saving overlay examples.
# Each entry is a tuple: (image_path, avg_confidence, overlay_image)
image_info_list = []

# ----------------------------
# Get list of all evaluation image files
# ----------------------------
image_files = glob.glob(eval_pattern)
print(f"Found {len(image_files)} images.")

# Optionally, set a font for annotation (if available)
try:
    font = ImageFont.truetype("arial.ttf", 15)
except:
    font = None

# ----------------------------
# Process each image
# ----------------------------
for img_path in tqdm(image_files, desc="Processing images"):
    # Load image and ensure RGB
    image = Image.open(img_path).convert("RGB")
    inputs = image_processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]

    # Filter detections for "car" and "person"
    filtered_indices = [
        i for i, label in enumerate(results["labels"])
        if model.config.id2label[label.item()] in desired_categories
    ]
    
    # Prepare lists for filtered detections on this image
    filtered_scores = [results["scores"][i] for i in filtered_indices]
    filtered_labels = [results["labels"][i] for i in filtered_indices]
    filtered_boxes = [results["boxes"][i] for i in filtered_indices]

    # Compute average confidence for the image.
    # If no filtered detection, assign 0.
    if filtered_scores:
        avg_conf = np.mean([s.item() for s in filtered_scores])
    else:
        avg_conf = 0.0

    # Create an overlay image copy and annotate detections
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    if filtered_scores:
        for score, label, box in zip(filtered_scores, filtered_labels, filtered_boxes):
            box_coords = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = box_coords
            text = f"{model.config.id2label[label.item()]}: {round(score.item(), 2)}"
            draw.rectangle((x, y, x2, y2), outline="red", width=2)
            # Use the specified font if available
            draw.text((x, y), text, fill="white", font=font)
            # Also aggregate detection data for plots
            all_detection_scores.append(score.item())
            all_detection_labels.append(model.config.id2label[label.item()])
            width = max(0, x2 - x)
            height = max(0, y2 - y)
            all_detection_areas.append(width * height)
    else:
        # If no detection, annotate image accordingly
        draw.text((10, 10), "No 'car' or 'person' detection", fill="yellow", font=font)
    
    # Save per-image info (for later selection) along with the overlay image
    image_info_list.append((img_path, avg_conf, overlay))

# ----------------------------
# Select images with lowest and highest average confidence
# ----------------------------
# Sort by average confidence
sorted_images = sorted(image_info_list, key=lambda x: x[1])
lowest_examples = sorted_images[:num_examples]
highest_examples = sorted_images[-num_examples:]

# Save the overlay images for these examples
for idx, (img_path, avg_conf, overlay_img) in enumerate(lowest_examples):
    base_name = os.path.basename(img_path).replace(".png", "")
    filename = os.path.join(output_dir, f"overlay_low_{idx+1}_{base_name}_avgConf_{avg_conf:.2f}.png")
    overlay_img.save(filename)
    print(f"Saved low confidence overlay: {filename}")

for idx, (img_path, avg_conf, overlay_img) in enumerate(highest_examples):
    base_name = os.path.basename(img_path).replace(".png", "")
    filename = os.path.join(output_dir, f"overlay_high_{idx+1}_{base_name}_avgConf_{avg_conf:.2f}.png")
    overlay_img.save(filename)
    print(f"Saved high confidence overlay: {filename}")

# ----------------------------
# Produce Composite Plots from All Images' Detections
# ----------------------------
# Check if there is at least one detection aggregated
if all_detection_scores:
    # Composite figure with three subplots:
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Histogram of Detection Scores
    axs[0].hist(all_detection_scores, bins=10, color='blue', edgecolor='black')
    axs[0].set_xlabel("Detection Confidence")
    axs[0].set_ylabel("Frequency")
    axs[0].set_title("Histogram of Detection Scores (Car, Person)")

    # Plot 2: Bar Chart of Detections per Category
    category_counts = Counter(all_detection_labels)
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    axs[1].bar(categories, counts, color='green', edgecolor='black')
    axs[1].set_xlabel("Category")
    axs[1].set_ylabel("Count")
    axs[1].set_title("Detections per Category (Car, Person)")
    axs[1].tick_params(axis='x', rotation=45)

    # Plot 3: Scatter Plot of Bounding Box Area vs. Detection Confidence
    axs[2].scatter(all_detection_areas, all_detection_scores, alpha=0.7, color='red')
    axs[2].set_xlabel("Bounding Box Area (pixels²)")
    axs[2].set_ylabel("Detection Confidence")
    axs[2].set_title("Area vs. Detection Confidence (Car, Person)")

    plt.tight_layout()
    composite_filename = os.path.join(output_dir, "combined_detection_plots_filtered.png")
    plt.savefig(composite_filename, dpi=300)
    print(f"Composite figure saved as {composite_filename}")
    plt.show()

    # ----------------------------
    # Save Individual Figures
    # ----------------------------
    # Histogram
    plt.figure(figsize=(6, 5))
    plt.hist(all_detection_scores, bins=10, color='blue', edgecolor='black')
    plt.xlabel("Detection Confidence")
    plt.ylabel("Frequency")
    plt.title("Histogram of Detection Scores (Car, Person)")
    plt.tight_layout()
    hist_filename = os.path.join(output_dir, "histogram_detection_scores_filtered.png")
    plt.savefig(hist_filename, dpi=300)
    print(f"Histogram saved as {hist_filename}")
    plt.close()

    # Bar Chart
    plt.figure(figsize=(6, 5))
    plt.bar(categories, counts, color='green', edgecolor='black')
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.title("Detections per Category (Car, Person)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    bar_filename = os.path.join(output_dir, "bar_chart_detections_per_category_filtered.png")
    plt.savefig(bar_filename, dpi=300)
    print(f"Bar chart saved as {bar_filename}")
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(all_detection_areas, all_detection_scores, alpha=0.7, color='red')
    plt.xlabel("Bounding Box Area (pixels²)")
    plt.ylabel("Detection Confidence")
    plt.title("Area vs. Detection Confidence (Car, Person)")
    plt.tight_layout()
    scatter_filename = os.path.join(output_dir, "scatter_plot_area_vs_confidence_filtered.png")
    plt.savefig(scatter_filename, dpi=300)
    print(f"Scatter plot saved as {scatter_filename}")
    plt.close()
else:
    print("No detections were aggregated from the images for plotting.")
