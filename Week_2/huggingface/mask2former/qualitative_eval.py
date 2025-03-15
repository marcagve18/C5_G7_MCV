import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from tqdm import tqdm
from collections import defaultdict

# Create an output folder if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Set device: use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the processor and model, then move the model to the chosen device
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance")
model.to(device)

def get_segmentation_results(image, processor, model, threshold=0.9):
    """
    Process an image and return the segmentation results.
    Moves input tensors to the GPU if available.
    """
    inputs = processor(images=image, return_tensors="pt")
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]], threshold=threshold
    )[0]
    return results

def plot_segmentation_overlay(image, results, model, output_path):
    """
    Overlays segmentation masks on the original image along with label annotations.
    Saves the resulting figure to output_path.
    """
    img_np = np.array(image)
    height, width = img_np.shape[:2]
    dpi = 100  # Adjust if needed

    # Create a figure with size matching the image dimensions exactly (in inches)
    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)

    # Show the original image
    ax.imshow(img_np)
    ax.axis('off')

    # Build a mapping from segment id to label id
    segment_to_label = {segment['id']: segment['label_id'] for segment in results["segments_info"]}

    # Overlay each segmentation mask with random colors and add text labels
    for segment in results['segments_info']:
        seg_id = segment['id']
        label = model.config.id2label[segment_to_label[seg_id]]
        
        # Generate a random RGB color for the overlay
        color = [random.random() for _ in range(3)]
        
        # Create a mask for this segment
        mask = (results['segmentation'].numpy() == seg_id)
        
        # Prepare a colored RGBA overlay for the mask
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
        colored_mask[mask, :3] = color
        colored_mask[mask, 3] = 0.9  # Set transparency
        
        # Overlay the colored mask onto the image
        ax.imshow(colored_mask, interpolation='none')
        
        # Calculate centroid of the mask to position the label
        ys, xs = np.where(mask)
        if len(ys) > 0 and len(xs) > 0:
            x_centroid = xs.mean()
            y_centroid = ys.mean()
            ax.text(
                x_centroid, y_centroid, f"#{seg_id}:{label}",
                color='white', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, pad=1)
            )
    
    # Save the visualization without extra padding
    plt.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)

def plot_aggregated_score_vs_area(areas, scores, annotations, output_path, annotate=True):
    """
    Plots a scatter chart of aggregated prediction scores versus area (pixel count)
    for all segmentation predictions, with a unique color assigned for each label.
    """
    # Determine unique labels and assign a color to each using the "tab10" colormap.
    unique_labels = sorted(set(annotations))
    colormap = plt.get_cmap("tab10")
    label_to_color = {label: colormap(i % colormap.N) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(8, 6))
    # Group points by label and plot each group with a label for the legend.
    for label in unique_labels:
        label_indices = [i for i, ann in enumerate(annotations) if ann == label]
        label_areas = [areas[i] for i in label_indices]
        label_scores = [scores[i] for i in label_indices]
        plt.scatter(label_areas, label_scores, alpha=0.7, color=label_to_color[label], label=label)

    plt.xlabel("Area (pixel count)")
    plt.ylabel("Score")
    plt.title("Aggregated Prediction Score vs. Area (Filtered)")
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_path)
    plt.close()

def get_label_counts(results, model):
    """
    Returns a dictionary with label frequencies for the segmentation results.
    """
    counts = {}
    for segment in results['segments_info']:
        label = model.config.id2label[segment['label_id']]
        counts[label] = counts.get(label, 0) + 1
    return counts

# New Plot Functions

def plot_average_confidence_per_class(annotations, scores, output_path):
    """
    Plots a bar chart of the average confidence per class.
    """
    class_scores = defaultdict(list)
    for ann, score in zip(annotations, scores):
        class_scores[ann].append(score)
    classes = sorted(class_scores.keys())
    avg_confidences = [sum(class_scores[c]) / len(class_scores[c]) for c in classes]
    
    plt.figure(figsize=(8, 6))
    plt.bar(classes, avg_confidences, color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Average Confidence")
    plt.title("Average Confidence per Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_average_area_per_class(annotations, areas, output_path):
    """
    Plots a bar chart of the average area per class.
    """
    class_areas = defaultdict(list)
    for ann, area in zip(annotations, areas):
        class_areas[ann].append(area)
    classes = sorted(class_areas.keys())
    avg_areas = [sum(class_areas[c]) / len(class_areas[c]) for c in classes]
    
    plt.figure(figsize=(8, 6))
    plt.bar(classes, avg_areas, color='coral')
    plt.xlabel("Class")
    plt.ylabel("Average Area (pixel count)")
    plt.title("Average Area per Class")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_score_distribution_per_class(annotations, scores, output_path):
    """
    Plots histograms of scores for each class in separate subplots.
    """
    class_scores = defaultdict(list)
    for ann, score in zip(annotations, scores):
        class_scores[ann].append(score)
    
    unique_classes = sorted(class_scores.keys())
    num_classes = len(unique_classes)
    fig, axs = plt.subplots(num_classes, 1, figsize=(8, 4 * num_classes))
    if num_classes == 1:
        axs = [axs]
    for ax, cls in zip(axs, unique_classes):
        ax.hist(class_scores[cls], bins=20, color='green', alpha=0.7)
        ax.set_title(f"Score Distribution for {cls}")
        ax.set_xlabel("Score")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_area_distribution_per_class(annotations, areas, output_path):
    """
    Plots histograms of area for each class in separate subplots.
    """
    class_areas = defaultdict(list)
    for ann, area in zip(annotations, areas):
        class_areas[ann].append(area)
    
    unique_classes = sorted(class_areas.keys())
    num_classes = len(unique_classes)
    fig, axs = plt.subplots(num_classes, 1, figsize=(8, 4 * num_classes))
    if num_classes == 1:
        axs = [axs]
    for ax, cls in zip(axs, unique_classes):
        ax.hist(class_areas[cls], bins=20, color='purple', alpha=0.7)
        ax.set_title(f"Area Distribution for {cls}")
        ax.set_xlabel("Area (pixel count)")
        ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# Instead of storing open image objects, store file paths
eval_img_paths = []
instances_path = "/ghome/c5mcv07/mcv/datasets/C5/KITTI-MOTS/testing/image_02"
for instance_id in os.listdir(instances_path):
    instance_folder = os.path.join(instances_path, instance_id)
    for file in os.listdir(instance_folder):
        if file.endswith(".png"):
            eval_img_paths.append(os.path.join(instance_folder, file))

# Define the allowed classes (in lowercase)
allowed_classes = ['car', 'person', 'pedestrian']

# Global lists to aggregate areas, scores, and annotations
aggregated_areas = []
aggregated_scores = []
aggregated_annotations = []
aggregated_counts = {}

# Process each image by opening it on-the-fly
for idx, img_path in enumerate(tqdm(eval_img_paths[:100])):
    with Image.open(img_path) as image:
        # Copy the image into memory so the file handle is released immediately
        image_copy = image.copy()
    
    # Get segmentation results using GPU
    results = get_segmentation_results(image_copy, processor, model, threshold=0.5)

    # Filter the results to only include allowed classes
    segments = results['segments_info']
    filtered_segments = []
    for s in segments:
        label = model.config.id2label[s['label_id']].lower()
        if label in allowed_classes:
            filtered_segments.append(s)
    results['segments_info'] = filtered_segments

    # (Optional) Save individual segmentation overlay image
    overlay_output_path = f"outputs/prediction_{idx}.png"
    # Uncomment the next line if you want to save overlays:
    # plot_segmentation_overlay(image_copy, results, model, overlay_output_path)
    print(f"Processed image {idx} (overlay saved to {overlay_output_path})")
    
    # Get the segmentation mask as a numpy array
    seg_mask = results['segmentation']
    if isinstance(seg_mask, torch.Tensor):
        seg_mask = seg_mask.cpu().numpy()

    # For each filtered segment, compute area and accumulate score and annotation
    for seg_info in results['segments_info']:
        seg_id = seg_info["id"]
        area = np.sum(seg_mask == seg_id)
        score = seg_info.get("score", None)
        if score is None:
            continue
        aggregated_areas.append(area)
        aggregated_scores.append(score)
        # Get annotation (label name)
        label = model.config.id2label.get(seg_info["label_id"], str(seg_info["label_id"]))
        aggregated_annotations.append(label)
    
    # (Optional) Count labels for this image and update aggregated counts
    label_counts = get_label_counts(results, model)
    for label, count in label_counts.items():
        aggregated_counts[label] = aggregated_counts.get(label, 0) + count

# After processing all images, generate and save the plots

# 1. Aggregated Scatter Plot (Score vs. Area)
scatter_output_path = "outputs/aggregated_score_vs_area.png"
plot_aggregated_score_vs_area(aggregated_areas, aggregated_scores, aggregated_annotations, scatter_output_path, annotate=True)
print(f"Aggregated scatter plot saved to {scatter_output_path}")

# 2. Aggregated Label Distribution
if aggregated_counts:
    plt.figure(figsize=(10, 6))
    labels = list(aggregated_counts.keys())
    counts = [aggregated_counts[label] for label in labels]
    plt.bar(labels, counts, color='teal')
    plt.xlabel("Segmentation Labels")
    plt.ylabel("Frequency")
    plt.title("Distribution of Segmentation Labels Across Images")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    distribution_chart_path = "outputs/label_distribution.png"
    plt.savefig(distribution_chart_path)
    plt.close()
    print(f"Saved aggregated label distribution chart to {distribution_chart_path}")
else:
    print("No labels found to aggregate.")

# 3. Average Confidence per Class
avg_conf_output_path = "outputs/avg_confidence_per_class.png"
plot_average_confidence_per_class(aggregated_annotations, aggregated_scores, avg_conf_output_path)
print(f"Average confidence per class chart saved to {avg_conf_output_path}")

# 4. Average Area per Class
avg_area_output_path = "outputs/avg_area_per_class.png"
plot_average_area_per_class(aggregated_annotations, aggregated_areas, avg_area_output_path)
print(f"Average area per class chart saved to {avg_area_output_path}")

# 5. Score Distribution per Class
score_hist_output_path = "outputs/score_distribution_per_class.png"
plot_score_distribution_per_class(aggregated_annotations, aggregated_scores, score_hist_output_path)
print(f"Score distribution per class chart saved to {score_hist_output_path}")

# 6. Area Distribution per Class
area_hist_output_path = "outputs/area_distribution_per_class.png"
plot_area_distribution_per_class(aggregated_annotations, aggregated_areas, area_hist_output_path)
print(f"Area distribution per class chart saved to {area_hist_output_path}")
