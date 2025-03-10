from ultralytics import YOLO
import torch

# Choose the device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLO model
model = YOLO("yolo11m.pt")

# Define dataset path
DATASET_PATH = "/home/c5mcv07/C5_G7_MCV/Week_1/yolov11/src/data.yaml"

# Define hyperparameter search space
search_space = {
    "lr0": (1e-6, 1e-2),         # Learning rate
    "momentum": (0.6, 0.98),     # Momentum for optimizer
    "weight_decay": (0.0, 0.01), # Regularization
    "lrf": (0.1, 0.5),           # Final learning rate fraction
    "degrees": (0.0, 45.0),      # Rotation augmentation
    "scale": (0.5, 2.0),         # Scale augmentation
    "shear": (0.0, 10.0),        # Shear transformation
}

# Run hyperparameter tuning
best_results = model.tune(
    data=DATASET_PATH,
    epochs=30,       # Number of epochs per tuning trial
    iterations=100,  # Number of hyperparameter trials
    optimizer="AdamW",
    space=search_space,
    plots=True,  # Enable plots for visualization
    save=True,   # Save best model and parameters
    val=True,    # Run validation after each trial
)

# Print best hyperparameters found
print("Best hyperparameters:", best_results)
