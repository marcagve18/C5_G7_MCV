from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import requests
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-large-coco-instance")
model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-large-coco-instance")

# Load the image
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Process the image and perform segmentation
inputs = processor(images=image, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# Get the segmentation results
results = processor.post_process_instance_segmentation(outputs, target_sizes=[image.size[::-1]], threshold=0.9)[0]
print(results.keys())
for segment in results['segments_info']:
    print(segment)

# Create a mapping from segment id to its label id
segment_to_label = {segment['id']: segment['label_id'] for segment in results["segments_info"]}
print(segment_to_label)

# Convert the original image to a numpy array for plotting
img_np = np.array(image)

# Assume img_np is your image as a numpy array
height, width = img_np.shape[:2]
dpi = 100  # adjust as needed

# Create a figure with the size exactly equal to the image size (in inches)
fig = plt.figure(figsize=(width/dpi, height/dpi), dpi=dpi)
# Remove margins by adjusting subplot parameters
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
# Create an axes that fills the whole figure
ax = plt.Axes(fig, [0, 0, 1, 1])
fig.add_axes(ax)

# Display the image
ax.imshow(img_np)
ax.axis('off')

# Overlay each segmentation mask and add label annotations
for segment in results['segments_info']:
    seg_id = segment['id']
    label = model.config.id2label[segment_to_label[seg_id]]
    
    # Generate a random RGB color
    color = [random.random() for _ in range(3)]
    
    # Create a boolean mask for this segment
    mask = (results['segmentation'].numpy() == seg_id)
    
    # Prepare a colored RGBA overlay for this mask
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 4))
    colored_mask[mask, :3] = color
    colored_mask[mask, 3] = 0.9  # Set transparency
    
    # Overlay the colored mask
    ax.imshow(colored_mask, interpolation='none')
    
    # Compute the centroid for the label placement
    ys, xs = np.where(mask)
    if len(ys) > 0 and len(xs) > 0:
        x_centroid = xs.mean()
        y_centroid = ys.mean()
        ax.text(x_centroid, y_centroid, f"#{seg_id}:{label}", color='white', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='black', alpha=0.5, pad=1))

# Save the figure without any white padding
plt.savefig("outputs/prediction.png", dpi=dpi, pad_inches=0)