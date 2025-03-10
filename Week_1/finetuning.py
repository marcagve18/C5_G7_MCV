import os
import torch
from ultralytics import YOLO

# Choose the device (GPU/CPU)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load the pretrained YOLO model
model = YOLO("yolo11m.pt")

# Define the dataset path and update data.yaml for KITTI MOTS dataset
DATASET_PATH = "/home/c5mcv07/C5_G7_MCV/Week_1/yolov11/src/data.yaml"

# Fine-tuning the model on the KITTI MOTS dataset
model.train(
    data=DATASET_PATH,
    epochs=50,
    batch=16,
    imgsz=1024,
    mosaic=0.8,
    hsv_h=0.01,  
    hsv_s=0.5,   
    hsv_v=0.3,         
    mixup=0.1    
)
print("Pere")

# 📊 Evaluate the fine-tuned model (NO NEED TO RELOAD!)
print("\n🔎 Running Evaluation on KITTI MOTS...")
results = model.val(data=DATASET_PATH, batch=8, device=device)

# Print specific metrics
print("Mean average precision at IoU=0.50:", results.box.map50)
print("Mean average precision at IoU=0.75:", results.box.map75)
print("Mean precision:", results.box.mp)
print("Mean recall:", results.box.mr)
