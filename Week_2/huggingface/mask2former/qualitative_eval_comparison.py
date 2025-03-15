import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import torch
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from tqdm import tqdm

# Create output directories
os.makedirs("outputs_baseline", exist_ok=True)
os.makedirs("outputs_finetuned", exist_ok=True)
os.makedirs("outputs_comparison", exist_ok=True)

# Set device: use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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

    fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    fig.add_axes(ax)
    ax.imshow(img_np)
    ax.axis('off')

    # Overlay each segmentation mask with a random color and add text label.
    segment_to_label = {segment['id']: segment['label_id'] for segment in results["segments_info"]}
    for segment in results['segments_info']:
        seg_id = segment['id']
        # Map the raw label
        raw_label = model.config.id2label[segment_to_label[seg_id]]
        label = map_label(raw_label)
        color = [random.random() for _ in range(3)]
        mask = (results['segmentation'].numpy() == seg_id)
        colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
        colored_mask[mask, :3] = color
        colored_mask[mask, 3] = 0.9
        ax.imshow(colored_mask, interpolation='none')
        ys, xs = np.where(mask)
        if len(ys) > 0 and len(xs) > 0:
            x_centroid = xs.mean()
            y_centroid = ys.mean()
            ax.text(x_centroid, y_centroid, f"#{seg_id}:{label}",
                    color='white', fontsize=12, ha='center', va='center',
                    bbox=dict(facecolor='black', alpha=0.5, pad=1))
    plt.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)

def plot_aggregated_score_vs_area(areas, scores, annotations, output_path, annotate=True):
    """
    Plots a scatter chart of aggregated prediction scores versus area (pixel count)
    for all segmentation predictions, with a unique color per label.
    """
    unique_labels = sorted(set(annotations))
    colormap = plt.get_cmap("tab10")
    label_to_color = {label: colormap(i % colormap.N) for i, label in enumerate(unique_labels)}
    plt.figure(figsize=(8, 6))
    # Group points by label for the legend.
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
    Applies mapping so that 'pedestrian' becomes 'person'.
    """
    counts = {}
    for segment in results['segments_info']:
        raw_label = model.config.id2label[segment['label_id']]
        label = map_label(raw_label)
        counts[label] = counts.get(label, 0) + 1
    return counts

# New plot functions for additional evaluation metrics

def plot_average_confidence_per_class(annotations, scores, output_path):
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

def run_evaluation(processor, model, eval_img_paths, threshold, allowed_classes, output_dir):
    """
    Runs evaluation on the provided image paths using the given processor and model.
    Saves individual plots in output_dir and returns aggregated metrics.
    """
    aggregated_areas = []
    aggregated_scores = []
    aggregated_annotations = []
    aggregated_counts = {}
    
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, img_path in enumerate(tqdm(eval_img_paths)):
        with Image.open(img_path) as image:
            image_copy = image.copy()
        results = get_segmentation_results(image_copy, processor, model, threshold=threshold)
        
        # Filter results to include only allowed classes (apply mapping to ensure 'pedestrian' becomes 'person')
        filtered_segments = []
        for s in results['segments_info']:
            raw_label = model.config.id2label[s['label_id']].lower()
            mapped_label = map_label(raw_label)
            if mapped_label in allowed_classes:
                filtered_segments.append(s)
        results['segments_info'] = filtered_segments

        # (Optional) Save overlay images
        overlay_output_path = os.path.join(output_dir, f"prediction_{idx}.png")
        # Uncomment the next line to save overlays:
        # plot_segmentation_overlay(image_copy, results, model, overlay_output_path)
        
        seg_mask = results['segmentation']
        if isinstance(seg_mask, torch.Tensor):
            seg_mask = seg_mask.cpu().numpy()

        for seg_info in results['segments_info']:
            seg_id = seg_info["id"]
            area = np.sum(seg_mask == seg_id)
            score = seg_info.get("score", None)
            if score is None:
                continue
            aggregated_areas.append(area)
            aggregated_scores.append(score)
            raw_label = model.config.id2label.get(seg_info["label_id"], str(seg_info["label_id"]))
            mapped_label = map_label(raw_label)
            aggregated_annotations.append(mapped_label)
        
        # Update label counts
        label_counts = get_label_counts(results, model)
        for label, count in label_counts.items():
            aggregated_counts[label] = aggregated_counts.get(label, 0) + count

    # Save aggregated plots for this model
    plot_aggregated_score_vs_area(aggregated_areas, aggregated_scores, aggregated_annotations,
                                  os.path.join(output_dir, "aggregated_score_vs_area.png"), annotate=True)
    plot_average_confidence_per_class(aggregated_annotations, aggregated_scores,
                                      os.path.join(output_dir, "avg_confidence_per_class.png"))
    plot_average_area_per_class(aggregated_annotations, aggregated_areas,
                                os.path.join(output_dir, "avg_area_per_class.png"))
    plot_score_distribution_per_class(aggregated_annotations, aggregated_scores,
                                      os.path.join(output_dir, "score_distribution_per_class.png"))
    plot_area_distribution_per_class(aggregated_annotations, aggregated_areas,
                                     os.path.join(output_dir, "area_distribution_per_class.png"))
    
    plt.figure(figsize=(10, 6))
    labels = list(aggregated_counts.keys())
    counts = [aggregated_counts[label] for label in labels]
    plt.bar(labels, counts, color='teal')
    plt.xlabel("Segmentation Labels")
    plt.ylabel("Frequency")
    plt.title("Distribution of Segmentation Labels")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "label_distribution.png"))
    plt.close()
    
    return {
        "areas": aggregated_areas,
        "scores": aggregated_scores,
        "annotations": aggregated_annotations,
        "counts": aggregated_counts,
    }

# ---------------------------
# Main: Evaluate both models
# ---------------------------

# Define the allowed classes (after mapping, 'pedestrian' becomes 'person')
# So we only need to include 'car' and 'person'
allowed_classes = ['car', 'person']

# Prepare image paths (adjust the directory as needed)
eval_img_paths = []
instances_path = "/ghome/c5mcv07/mcv/datasets/C5/KITTI-MOTS/testing/image_02"
for instance_id in os.listdir(instances_path):
    instance_folder = os.path.join(instances_path, instance_id)
    for file in os.listdir(instance_folder):
        if file.endswith(".png"):
            eval_img_paths.append(os.path.join(instance_folder, file))
eval_img_paths = eval_img_paths[:100]

# Load the baseline model and its processor
baseline_processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
baseline_model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance")
baseline_model.to(device)

# Load the finetuned model and its processor (adjust the paths accordingly)
finetuned_processor = AutoImageProcessor.from_pretrained("/ghome/c5mcv07/C5_G7_MCV/Week_2/huggingface/mask2former/checkpoints/Mask2Former_KITTI_v1/checkpoint-2000")
finetuned_model = Mask2FormerForUniversalSegmentation.from_pretrained("/ghome/c5mcv07/C5_G7_MCV/Week_2/huggingface/mask2former/checkpoints/Mask2Former_KITTI_v1/checkpoint-2000")
finetuned_model.to(device)

# Run evaluations
baseline_metrics = run_evaluation(baseline_processor, baseline_model, eval_img_paths, threshold=0.5,
                                  allowed_classes=allowed_classes, output_dir="outputs_baseline")
finetuned_metrics = run_evaluation(finetuned_processor, finetuned_model, eval_img_paths, threshold=0.5,
                                   allowed_classes=allowed_classes, output_dir="outputs_finetuned")

# ---------------------------
# Comparative Plots
# ---------------------------

# 1. Combined Aggregated Scatter Plot (Score vs. Area)
plt.figure(figsize=(8,6))
plt.scatter(baseline_metrics["areas"], baseline_metrics["scores"],
            color="blue", marker="o", alpha=0.6, label="Baseline")
plt.scatter(finetuned_metrics["areas"], finetuned_metrics["scores"],
            color="red", marker="x", alpha=0.6, label="Finetuned")
plt.xlabel("Area (pixel count)")
plt.ylabel("Score")
plt.title("Comparison: Aggregated Score vs. Area")
plt.legend()
plt.tight_layout()
plt.savefig("outputs_comparison/aggregated_score_vs_area_comparison.png")
plt.close()
print("Saved aggregated scatter plot comparison to outputs_comparison/aggregated_score_vs_area_comparison.png")

# 2. Average Confidence per Class Comparison
def compute_average_confidence(annotations, scores):
    class_scores = defaultdict(list)
    for ann, score in zip(annotations, scores):
        class_scores[ann].append(score)
    return {cls: sum(scores_list) / len(scores_list) for cls, scores_list in class_scores.items()}

baseline_avg_conf = compute_average_confidence(baseline_metrics["annotations"], baseline_metrics["scores"])
finetuned_avg_conf = compute_average_confidence(finetuned_metrics["annotations"], finetuned_metrics["scores"])

classes = sorted(set(list(baseline_avg_conf.keys()) + list(finetuned_avg_conf.keys())))
baseline_conf = [baseline_avg_conf.get(cls, 0) for cls in classes]
finetuned_conf = [finetuned_avg_conf.get(cls, 0) for cls in classes]

x = np.arange(len(classes))
width = 0.35
plt.figure(figsize=(8,6))
plt.bar(x - width/2, baseline_conf, width, label='Baseline', color='blue')
plt.bar(x + width/2, finetuned_conf, width, label='Finetuned', color='red')
plt.xlabel("Class")
plt.ylabel("Average Confidence")
plt.title("Average Confidence per Class Comparison")
plt.xticks(x, classes, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("outputs_comparison/avg_confidence_comparison.png")
plt.close()
print("Saved average confidence comparison to outputs_comparison/avg_confidence_comparison.png")

# 3. Average Area per Class Comparison
def compute_average_area(annotations, areas):
    class_areas = defaultdict(list)
    for ann, area in zip(annotations, areas):
        class_areas[ann].append(area)
    return {cls: sum(a_list) / len(a_list) for cls, a_list in class_areas.items()}

baseline_avg_area = compute_average_area(baseline_metrics["annotations"], baseline_metrics["areas"])
finetuned_avg_area = compute_average_area(finetuned_metrics["annotations"], finetuned_metrics["areas"])

classes = sorted(set(list(baseline_avg_area.keys()) + list(finetuned_avg_area.keys())))
baseline_area = [baseline_avg_area.get(cls, 0) for cls in classes]
finetuned_area = [finetuned_avg_area.get(cls, 0) for cls in classes]

x = np.arange(len(classes))
width = 0.35
plt.figure(figsize=(8,6))
plt.bar(x - width/2, baseline_area, width, label='Baseline', color='blue')
plt.bar(x + width/2, finetuned_area, width, label='Finetuned', color='red')
plt.xlabel("Class")
plt.ylabel("Average Area (pixel count)")
plt.title("Average Area per Class Comparison")
plt.xticks(x, classes, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("outputs_comparison/avg_area_comparison.png")
plt.close()
print("Saved average area comparison to outputs_comparison/avg_area_comparison.png")

# 4. Aggregated Label Distribution Comparison (Side-by-Side Bar Chart)
baseline_counts = baseline_metrics["counts"]
finetuned_counts = finetuned_metrics["counts"]
all_classes = sorted(set(list(baseline_counts.keys()) + list(finetuned_counts.keys())))
baseline_vals = [baseline_counts.get(cls, 0) for cls in all_classes]
finetuned_vals = [finetuned_counts.get(cls, 0) for cls in all_classes]

x = np.arange(len(all_classes))
width = 0.35
plt.figure(figsize=(10,6))
plt.bar(x - width/2, baseline_vals, width, label='Baseline', color='blue')
plt.bar(x + width/2, finetuned_vals, width, label='Finetuned', color='red')
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Label Distribution Comparison")
plt.xticks(x, all_classes, rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig("outputs_comparison/label_distribution_comparison.png")
plt.close()
print("Saved label distribution comparison to outputs_comparison/label_distribution_comparison.png")
