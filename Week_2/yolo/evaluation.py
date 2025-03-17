import torch
from ultralytics import YOLO

# Select device (GPU if available)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv11 segmentation model (use the correct pretrained model)
model = YOLO("yolo11m-seg.pt")  # Ensure this is a segmentation model

# Define dataset path
DATASET_PATH = "/home/c5mcv07/C5_G7_MCV/Week_2/yolo/src/data.yaml"

# Run validation
metrics = model.val(data=DATASET_PATH, task="segment", classes=[0, 2])  # task="segment" for segmentation

# Extract specific results

print("Mean average precision:",metrics.seg.map)  # map50-95(M)
print("Mean average precision at IoU=0.50:",metrics.seg.map50)  # map50(M)
print("Mean average precision at IoU=0.75:",metrics.seg.map75)  # map75(M)
print(metrics.seg.maps)