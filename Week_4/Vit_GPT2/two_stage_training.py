import wandb
from transformers import (
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback
)
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import shutil # For potentially cleaning up checkpoints later

# --- Assumed Utility Imports ---
# Make sure these paths are correct and files exist
sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")
try:
    from Image_Captioning_Utils.dataset import FoodDatasetWord
    from Image_Captioning_Utils.utils import get_train_val_test_annotations_split
    from Image_Captioning_Utils.metrics import calculate_metrics # Ensure this exists
except ImportError as e:
    print(f"Error importing utility functions: {e}")
    print("Please ensure Image_Captioning_Utils package is correctly installed and accessible.")
    sys.exit(1)
# --- End Utility Imports ---

import numpy as np
from dotenv import load_dotenv

load_dotenv()
SEED = 42

# Set seeds for Python, NumPy, and PyTorch
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Configuration ---
WANDB_KEY = os.getenv('WANDB_MARC')
WANDB_PROJECT = "C5_W4_ViT-GPT2" # Project name for Wandb

BASE_OUTPUT_DIR = "checkpoints/vit-gpt2-food-captioning-two-stage"
INITIAL_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"
TEXT_MAX_LEN = 16 # Consider if this is appropriate for food captions
METRIC_TO_OPTIMIZE = "bleu1" # Choose 'bleu4', 'rougeL', etc. Make sure compute_metrics returns it!

# Stage 1 Config: Train Decoder only
stage1_config = {
    "lr": 1e-5, # Often higher LR works when only training decoder
    "batch_size": 40,
    "epochs": 10, # Adjust as needed, early stopping will help
    "weight_decay": 1e-5,
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 50,
    "run_name": "stage1_train_decoder",
    "output_dir": os.path.join(BASE_OUTPUT_DIR, "stage1_decoder_only"),
    "num_workers": 8,
    "save_total_limit": 3 # Keep top 3 checkpoints + the best one
}

# Stage 2 Config: Train All with lower LR
stage2_config = {
    "lr": 1e-6, # Lower learning rate for full fine-tuning
    "batch_size": 40, # Can sometimes reduce if GPU memory becomes an issue
    "epochs": 10, # Adjust as needed
    "weight_decay": 1e-5,
    "eval_steps": 100,
    "save_steps": 100,
    "logging_steps": 50,
    "run_name": "stage2_train_all",
    "output_dir": os.path.join(BASE_OUTPUT_DIR, "stage2_full_finetune"),
    "num_workers": 8,
    "save_total_limit": 3
}

# Generation config
gen_kwargs = {"max_length": TEXT_MAX_LEN, "num_beams": 4}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Data Preparation ---
print("Loading and splitting data...")
splits = get_train_val_test_annotations_split()
train_annotations = splits["train"]
val_annotations = splits["val"]

train_dataset = FoodDatasetWord(train_annotations)
val_dataset = FoodDatasetWord(val_annotations)
print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Load tokenizer and feature extractor *once*
print(f"Loading tokenizer and feature extractor from {INITIAL_MODEL_NAME}...")
feature_extractor = ViTImageProcessor.from_pretrained(INITIAL_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(INITIAL_MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Set tokenizer pad_token to eos_token: {tokenizer.pad_token}")

# Collate function
def custom_collate_fn(batch):
    images, captions = zip(*batch)
    pixel_values = feature_extractor(images=list(images), return_tensors="pt").pixel_values
    tokenized = tokenizer(
        list(captions),
        padding="max_length",
        truncation=True,
        max_length=TEXT_MAX_LEN,
        return_tensors="pt",
    )
    # Replace padding token id in labels with -100 to ignore loss calculation
    labels = tokenized["input_ids"].clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

# Metrics function
def compute_metrics_func(eval_pred):
    preds = eval_pred.predictions
    preds = np.where(preds == -100, tokenizer.pad_token_id, preds)
    labels = eval_pred.label_ids
    labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

    # Decode the generated predictions and the ground truth labels
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    from Image_Captioning_Utils.metrics import calculate_metrics
    metrics = calculate_metrics(decoded_labels, decoded_preds)
    # You can log additional info if needed
    for metric, score in metrics.items():
        print(f"{metric}: {score}")
    return metrics

# ==================================
#          STAGE 1: Train Decoder
# ==================================
print("\n" + "="*30)
print("         STAGE 1: Training Decoder")
print("="*30)

# Initialize Wandb for Stage 1
wandb.login(key=WANDB_KEY)
wandb.init(
    project=WANDB_PROJECT,
    name=stage1_config["run_name"],
    config=stage1_config,
    reinit=True # Allows initializing multiple runs in one script
)

# Load the initial pre-trained model
print(f"Loading initial model: {INITIAL_MODEL_NAME}")
model_stage1 = VisionEncoderDecoderModel.from_pretrained(INITIAL_MODEL_NAME)
model_stage1.to(device)

# Configure model for generation (important for predict_with_generate)
model_stage1.config.eos_token_id = tokenizer.eos_token_id
model_stage1.config.decoder_start_token_id = tokenizer.bos_token_id
model_stage1.config.pad_token_id = tokenizer.pad_token_id

# Freeze encoder parameters
print("Freezing ENCODER parameters for Stage 1...")
for param in model_stage1.encoder.parameters():
    param.requires_grad = False
# Ensure decoder parameters are trainable (should be by default, but good practice)
for param in model_stage1.decoder.parameters():
    param.requires_grad = True

num_trainable_params_s1 = sum(p.numel() for p in model_stage1.parameters() if p.requires_grad)
total_params_s1 = sum(p.numel() for p in model_stage1.parameters())
print(f"Stage 1 - Trainable parameters: {num_trainable_params_s1} / {total_params_s1} ({num_trainable_params_s1/total_params_s1*100:.2f}%)")

# Define Training Arguments for Stage 1
training_args_stage1 = Seq2SeqTrainingArguments(
    output_dir=stage1_config["output_dir"],
    learning_rate=stage1_config['lr'],
    num_train_epochs=stage1_config['epochs'],
    per_device_train_batch_size=stage1_config['batch_size'],
    per_device_eval_batch_size=stage1_config['batch_size'],
    weight_decay=stage1_config['weight_decay'],
    dataloader_num_workers=stage1_config['num_workers'],
    eval_strategy="steps",
    eval_steps=stage1_config['eval_steps'],
    logging_steps=stage1_config['logging_steps'],
    save_steps=stage1_config['save_steps'],
    predict_with_generate=True, # Crucial for metrics calculation
    logging_dir=f"{stage1_config['output_dir']}/logs",
    save_total_limit=stage1_config['save_total_limit'],
    load_best_model_at_end=True, # <<< Load the best model based on metric
    metric_for_best_model=METRIC_TO_OPTIMIZE, # <<< Your key metric
    greater_is_better=True, # <<< Assuming higher Bleu/Rouge is better
    report_to="wandb",
    #gradient_accumulation_steps=2 # Adjust if batch size is too large for memory
)

# Define Trainer for Stage 1
trainer_stage1 = Seq2SeqTrainer(
    model=model_stage1,
    args=training_args_stage1,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_collate_fn,
    compute_metrics=compute_metrics_func,
    tokenizer=feature_extractor, # Pass tokenizer for generation inside trainer
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # Stop if no improvement for 5 evaluations
)

# Train Stage 1
print("Starting Stage 1 training...")
train_result_s1 = trainer_stage1.train()
print("Stage 1 training finished.")

# Save metrics and state
trainer_stage1.log_metrics("train", train_result_s1.metrics)
trainer_stage1.save_metrics("train", train_result_s1.metrics)
trainer_stage1.save_state()

# Explicitly save the best model found during Stage 1 training
best_model_s1_path = os.path.join(stage1_config["output_dir"], "best_model_stage1")
print(f"Saving the best model from Stage 1 to: {best_model_s1_path}")
trainer_stage1.save_model(best_model_s1_path) # The trainer loaded the best model automatically

# Evaluate the best model from Stage 1 one last time (optional but good)
print("Evaluating the best model from Stage 1...")
eval_metrics_s1 = trainer_stage1.evaluate()
print(f"Best Stage 1 Model Evaluation Metrics ({METRIC_TO_OPTIMIZE}): {eval_metrics_s1}")
trainer_stage1.log_metrics("eval_best", eval_metrics_s1)
trainer_stage1.save_metrics("eval_best", eval_metrics_s1)

wandb.finish() # End the Wandb run for Stage 1

# Clean up model variable to free memory if needed
del model_stage1
del trainer_stage1
torch.cuda.empty_cache() # Clear GPU cache

# ==================================
#          STAGE 2: Train All
# ==================================
print("\n" + "="*30)
print("         STAGE 2: Training All (Fine-tuning)")
print("="*30)

# Initialize Wandb for Stage 2
wandb.init(
    project=WANDB_PROJECT,
    name=stage2_config["run_name"],
    config=stage2_config,
    reinit=True
)

# Load the best model saved from Stage 1
print(f"Loading best model from Stage 1: {best_model_s1_path}")
model_stage2 = VisionEncoderDecoderModel.from_pretrained(best_model_s1_path)
model_stage2.to(device)

# Re-configure model for generation (just in case config wasn't saved perfectly)
model_stage2.config.eos_token_id = tokenizer.eos_token_id
model_stage2.config.decoder_start_token_id = tokenizer.bos_token_id
model_stage2.config.pad_token_id = tokenizer.pad_token_id

# Unfreeze ALL parameters for Stage 2
print("Unfreezing ALL parameters for Stage 2...")
for param in model_stage2.parameters():
    param.requires_grad = True

num_trainable_params_s2 = sum(p.numel() for p in model_stage2.parameters() if p.requires_grad)
total_params_s2 = sum(p.numel() for p in model_stage2.parameters())
print(f"Stage 2 - Trainable parameters: {num_trainable_params_s2} / {total_params_s2} ({num_trainable_params_s2/total_params_s2*100:.2f}%)")


# Define Training Arguments for Stage 2 (note the lower learning rate)
training_args_stage2 = Seq2SeqTrainingArguments(
    output_dir=stage2_config["output_dir"],
    learning_rate=stage2_config['lr'], # <<< Lower LR
    num_train_epochs=stage2_config['epochs'],
    per_device_train_batch_size=stage2_config['batch_size'],
    per_device_eval_batch_size=stage2_config['batch_size'],
    weight_decay=stage2_config['weight_decay'],
    dataloader_num_workers=stage2_config['num_workers'],
    eval_strategy="steps",
    eval_steps=stage2_config['eval_steps'],
    logging_steps=stage2_config['logging_steps'],
    save_steps=stage2_config['save_steps'],
    predict_with_generate=True,
    logging_dir=f"{stage2_config['output_dir']}/logs",
    save_total_limit=stage2_config['save_total_limit'],
    load_best_model_at_end=True,
    metric_for_best_model=METRIC_TO_OPTIMIZE,
    greater_is_better=True,
    report_to="wandb",
    #gradient_accumulation_steps=2
)

# Define Trainer for Stage 2
trainer_stage2 = Seq2SeqTrainer(
    model=model_stage2,
    args=training_args_stage2,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=custom_collate_fn,
    compute_metrics=compute_metrics_func,
    tokenizer=feature_extractor,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # Early stopping for stage 2 as well
)

# Train Stage 2
print("Starting Stage 2 training...")
train_result_s2 = trainer_stage2.train() # Training starts from the loaded best Stage 1 model
print("Stage 2 training finished.")

# Save metrics and state for Stage 2
trainer_stage2.log_metrics("train", train_result_s2.metrics)
trainer_stage2.save_metrics("train", train_result_s2.metrics)
trainer_stage2.save_state()

# Explicitly save the final best model from Stage 2 training
final_best_model_path = os.path.join(stage2_config["output_dir"], "best_model_final")
print(f"Saving the best model from Stage 2 to: {final_best_model_path}")
trainer_stage2.save_model(final_best_model_path)

# Evaluate the final best model
print("Evaluating the final best model from Stage 2...")
eval_metrics_s2 = trainer_stage2.evaluate()
print(f"Final Best Model Evaluation Metrics ({METRIC_TO_OPTIMIZE}): {eval_metrics_s2}")
trainer_stage2.log_metrics("eval_best", eval_metrics_s2)
trainer_stage2.save_metrics("eval_best", eval_metrics_s2)

wandb.finish() # End the Wandb run for Stage 2

print("\nTwo-stage training complete!")
print(f"Best model from Stage 1 saved at: {best_model_s1_path}")
print(f"Best model from Stage 2 saved at: {final_best_model_path}")

# Optional: Clean up intermediate checkpoints if desired, keeping only the best ones
# print("Cleaning up intermediate checkpoints...")
# for stage_dir in [stage1_config["output_dir"], stage2_config["output_dir"]]:
#     for item in os.listdir(stage_dir):
#         item_path = os.path.join(stage_dir, item)
#         if os.path.isdir(item_path) and item.startswith("checkpoint-"):
#             print(f" Removing intermediate checkpoint: {item_path}")
#             shutil.rmtree(item_path)