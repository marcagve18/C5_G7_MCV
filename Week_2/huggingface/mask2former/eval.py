import os
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_utils

# Paths (adjust these to your dataset)
gt_annotation_file = "/ghome/c5mcv07/C5_G7_MCV/Week_2/huggingface/data_processing/test_annotations.json"  # COCO ground truth annotations
image_dir = "/home/mcv/datasets/C5/KITTI-MOTS/training/image_02"  # directory with validation images

# Set up device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#checkpoint = "facebook/mask2former-swin-tiny-coco-instance"
checkpoint = "/ghome/c5mcv07/C5_G7_MCV/Week_2/huggingface/mask2former/checkpoints/Mask2Former_KITTI_v1/checkpoint-2000"
# Load the processor and model
processor = AutoImageProcessor.from_pretrained(checkpoint)
model = Mask2FormerForUniversalSegmentation.from_pretrained(checkpoint)
model.to(device)
model.eval()  # set model to evaluation mode

# Load COCO ground truth
coco_gt = COCO(gt_annotation_file)

predictions = []

# Iterate over all images in the COCO ground truth
img_ids = coco_gt.getImgIds()
for image_info in tqdm(coco_gt.loadImgs(img_ids), desc="Evaluating"):
    image_id = image_info["id"]
    file_name = image_info["file_name"]
    image_path = os.path.join(image_dir, file_name)
    
    # Load image in RGB
    image = Image.open(image_path).convert("RGB")
    
    # Process image and run inference
    inputs = processor(images=image, return_tensors="pt")
    # Move tensors to the device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process to obtain instance segmentation results.
    # Note: target_sizes expects (height, width) of the original image.
    results = processor.post_process_instance_segmentation(
        outputs, target_sizes=[image.size[::-1]], threshold=0.9
    )[0]
    
    # 'results' contains:
    #   - 'segmentation': a 2D numpy array mapping each pixel to a segment id
    #   - 'segments_info': list of dicts with info (id, label_id, score, etc.)
    seg_map = results["segmentation"].numpy()
    for segment in results["segments_info"]:
        seg_id = segment["id"]
        category_id = segment["label_id"]
        score = segment["score"]
        
        # Create a binary mask for this instance
        binary_mask = (seg_map == seg_id).astype(np.uint8)
        if binary_mask.sum() == 0:
            continue
        
        # Encode the mask using pycocotools (ensure Fortran order)
        encoded_mask = mask_utils.encode(np.asfortranarray(binary_mask))
        # COCO expects the 'counts' field as a string, not bytes
        encoded_mask["counts"] = encoded_mask["counts"].decode("utf-8")
        
        if model.config.id2label[category_id] not in ["car", "person"]:
            print(f"skipping {model.config.id2label[category_id]}")
            continue
        else:
            print(f"adding {model.config.id2label[category_id]}")
        
        print(model.config.label2id)
        #model_person_category = model.config.label2id["person"]
        model_person_category = model.config.label2id["pedestrian"]
        model_car_category = model.config.label2id["car"]

        category_map = {
            model_car_category: 1,
            model_person_category: 2
        }

        prediction = {
            "image_id": image_id,
            "category_id": category_map[category_id],
            "segmentation": encoded_mask,
            "score": score
        }
        print(prediction)
        predictions.append(prediction)

# Save predictions to a JSON file
pred_file = "predictions.json"
with open(pred_file, "w") as f:
    json.dump(predictions, f)

# Run COCO evaluation
coco_dt = coco_gt.loadRes(pred_file)
coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
