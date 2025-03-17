import optuna
from ultralytics import YOLO
import torch

torch.cuda.empty_cache()

DATASET_PATH = "/home/c5mcv07/C5_G7_MCV/Week_2/yolo/src/data.yaml"

# Define the Optuna objective function
def objective(trial):
    # Hyperparameters to optimize
    lr0 = trial.suggest_loguniform('lr0', 1e-6, 1e-2)  # Initial learning rate
    lrf = trial.suggest_uniform('lrf', 0.1, 1.0)  # Learning rate factor
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    momentum = trial.suggest_uniform('momentum', 0.85, 0.95)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-6, 1e-3)

    # Load YOLO model
    model = YOLO("yolo11m-seg.pt")  # Load the pretrained model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on device: {device}")
    
    # Train with the suggested hyperparameters
    model.train(data="/home/c5mcv07/C5_G7_MCV/Week_2/yolo/src/data.yaml", epochs=50, batch=batch_size, lr0=lr0, lrf=lrf,
                momentum=momentum, weight_decay=weight_decay, imgsz=640, device = device)

    # Evaluate model performance on validation set
    results = model.val()  # Evaluate on validation set
    return results.seg.map  # Return mAP50-95 for optimization
# Optimize hyperparameters using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # You can change the number of trials

# Print the best hyperparameters found
print("Best hyperparameters:", study.best_params)
