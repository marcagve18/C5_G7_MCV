from data_processing import get_hf_dataset
from transformers import AutoImageProcessor, DetrForObjectDetection
import torch 
from PIL import Image, ImageDraw

image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

image = Image.open('/ghome/c5mcv07/mcv/datasets/C5/KITTI-MOTS/training/image_02/0000/000000.png')
inputs = image_processor(images=image, return_tensors="pt")

outputs = model(**inputs)

# convert outputs (bounding boxes and class logits) to Pascal VOC format (xmin, ymin, xmax, ymax)
target_sizes = torch.tensor([image.size[::-1]])
results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[
    0
]

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
        f"{round(score.item(), 3)} at location {box}"
    )

im1 = image.copy()
draw = ImageDraw.Draw(im1)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    draw.text((x, y), model.config.id2label[label.item()], fill="white")

im1.save("outputs/output_detr_inference_all_categories.png")

im2 = image.copy()
draw = ImageDraw.Draw(im2)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    category_name = model.config.id2label[label.item()]
    if category_name in ["car", "person"]:
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)
        draw.text((x, y), model.config.id2label[label.item()], fill="white")

im2.save("outputs/output_detr_inference_filtered_categories.png")
