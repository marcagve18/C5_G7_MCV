import os
import cv2
import numpy as np
from pycocotools.mask import decode
from ultralytics import YOLO
import torch
import optuna

# Check for GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Define dataset path
DATASET_PATH = "/home/c5mcv07/C5_G7_MCV/Week_1/yolov11/src/data.yaml"

def objective(trial):
    """
    Optuna optimization function to fine-tune YOLO hyperparameters for KITTI-MOTS.
    """
    # Adjusted hyperparameter search space
    lr0 = trial.suggest_loguniform("lr0", 1e-4, 5e-3)  # Adjusted learning rate
    lrf = trial.suggest_uniform("lrf", 0.1, 0.4)  # Final LR factor
    batch_size = trial.suggest_categorical("batch", [4, 8, 16])  # Adjusted batch size
    img_size = trial.suggest_categorical("imgsz", [1024, 1280, 1536])  # Higher resolution
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 5e-4)  # Adjusted weight decay
    momentum = trial.suggest_uniform("momentum", 0.85, 0.95)  # Momentum

    # Load YOLO model
    model = YOLO("yolo11m.pt").to(device)

    # Train model with the selected hyperparameters
    results = model.train(
        data=DATASET_PATH,
        epochs=30,
        batch=batch_size,
        imgsz=img_size,
        lr0=lr0,
        lrf=lrf,
        weight_decay=weight_decay,
        momentum=momentum,
        device=device,
        amp=True,  # Mixed precision training for faster computation
        patience=10  # Stops early if no improvement
    )

    # Return validation mean average precision (mAP) for Optuna to optimize
    return results.box.map

# Run Optuna optimization
study = optuna.create_study(direction="minimize")  # Minimize validation loss
study.optimize(objective, n_trials=10)  # Run 20 trials

# Print best hyperparameters
print("Best hyperparameters found by Optuna:", study.best_params)

# Load best parameters and retrain the model
best_params = study.best_params
model = YOLO("yolo11m.pt").to(device)
model.train(
    data=DATASET_PATH,
    epochs=50,  # Train longer with best params
    batch=best_params["batch"],
    imgsz=best_params["imgsz"],
    lr0=best_params["lr0"],
    lrf=best_params["lrf"],
    weight_decay=best_params["weight_decay"],
    momentum=best_params["momentum"],
    device=device,
    amp=True
)

# Run validation with fine-tuned model
results = model.val(data=DATASET_PATH)

# Print final validation metrics
print("Fine-tuned model validation results:")
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