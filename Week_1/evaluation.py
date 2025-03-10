import os
import cv2
import numpy as np
from pycocotools.mask import decode
from ultralytics import YOLO
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = YOLO("yolo11m.pt")

DATASET_PATH = "/home/c5mcv07/C5_G7_MCV/Week_1/yolov11/src/data.yaml"

# Run validation
results = model.val(data=DATASET_PATH, classes=[0, 2])

allowed_classes = [0, 2]
filtered_results = {cls: results.box.all_ap[i] for i, cls in results.names.items() if cls in allowed_classes}

print("Filtered AP results:", filtered_results)

# Print specific metrics
print("Classes: ", results.names)
print("Average precision for all classes:", results.box.all_ap)
print("Average precision:", results.box.ap)
print("Average precision at IoU=0.50:", results.box.ap50)
print("F1 score:", results.box.f1)
print("Mean average precision:", results.box.map)
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)
print("Mean precision:", results.box.mp)
print("Mean recall:", results.box.mr)
print("Precision:", results.box.p)
print("Recall:", results.box.r)


