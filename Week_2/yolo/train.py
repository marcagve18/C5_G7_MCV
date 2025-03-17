from ultralytics import YOLO
import torch

# Load the YOLO model
model = YOLO("yolo11m-seg.pt")  # Ensure this is a segmentation model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on device: {device}")

# Define dataset path
DATASET_PATH = "/home/c5mcv07/C5_G7_MCV/Week_2/yolo/src/data.yaml"

# Define best hyperparameters
best_hyperparameters = {
    "lr0": 1e-3,
    "lrf": 0.9,  # Adjusted for stability
    "batch": 0.8,
    "imgsz": 1024,
    "momentum": 0.90,  # More stability
    "weight_decay": 0.0005  # Medium regularization
}

# Train the model and capture metrics
results = model.train(
    data=DATASET_PATH,
    epochs=100,
    **best_hyperparameters,
    device=device,
    patience = 10,
    optimizer = 'SGD',
    verbose=True  # Show loss updates
)

model_path = "yolo_finetuned_model.pt"  # Replace with your desired path
model.save(model_path)

# Show loss and training metrics
# Extract training loss and mAP from results
print("\nTraining Metrics:")
print(f"Mean Average Precision (mAP50-95): {results.seg.map:.4f}")
print(f"Mean Average Precision at IoU=0.50: {results.seg.map50:.4f}")
print(f"Mean Average Precision at IoU=0.75: {results.seg.map75:.4f}")
# Evaluate the model on validation set
metrics = model.val(data=DATASET_PATH, task="segment", classes=[0, 2])

# Print validation metrics
print("\nValidation Metrics:")
print(f"Mean Average Precision (mAP50-95): {metrics.seg.map:.4f}")
print(f"Mean Average Precision at IoU=0.50: {metrics.seg.map50:.4f}")
print(f"Mean Average Precision at IoU=0.75: {metrics.seg.map75:.4f}")
print
