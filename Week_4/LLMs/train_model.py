# Standard library imports
import os
import sys
from datetime import datetime

# Third-party imports
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    ViTImageProcessor, 
    ViTModel,
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from dotenv import load_dotenv
import wandb
from tqdm import tqdm

# Local imports
sys.path.insert(0, "/ghome/c5mcv07/C5_G7_MCV")
from Image_Captioning_Utils.metrics import calculate_metrics
from Image_Captioning_Utils.dataset import FoodDatasetWord, FoodDatasetWordLevel
from Image_Captioning_Utils.utils import get_train_val_test_annotations_split
from Image_Captioning_Utils.constants import TEXT_MAX_LEN, SUBWORD_MAX_LEN, NUM_WORDS, WORD2IDX, IDX2WORD


# Configuration and Setup
load_dotenv()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
device = "cuda" if torch.cuda.is_available() else "cpu"

class Projection(nn.Module):
    def __init__(self, vit_hidden_size=768, llama_hidden_size=4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(vit_hidden_size, llama_hidden_size),
            nn.GELU(),
            nn.LayerNorm(llama_hidden_size)
        )

    def forward(self, x):
        return self.proj(x)

def initialize_models(llama_model_name: str = "meta-llama/Llama-3.2-1B"):
    # ViT Model
    print("Initializing ViT model...")
    vit_model_name = "google/vit-base-patch16-224-in21k"
    processor = ViTImageProcessor.from_pretrained(vit_model_name)
    vit_model = ViTModel.from_pretrained(vit_model_name).to(device)
    
    # Freeze ViT parameters
    for param in vit_model.parameters():
        param.requires_grad = False
        
    # Llama Model with LoRA
    print("Initializing Llama model with LoRA...")
    #tokenizer = AutoTokenizer.from_pretrained(llama_model_name)
    #tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        llama_model_name,
        device_map=device,
        torch_dtype=torch.float32
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # New custom embedding layer
    print("Initializing custom word-level embedding layer...")
    embedding_layer = nn.Embedding(NUM_WORDS, model.config.hidden_size).to(device)
    
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    
    # Projection layer
    print("Initializing projection layer...")
    projection_layer = Projection(
        vit_hidden_size=768,
        llama_hidden_size=model.config.hidden_size
    ).to(device)
    
    return processor, vit_model, peft_model, projection_layer, embedding_layer #,tokenizer

def create_data_loaders(processor, training_config):
    def custom_collate_fn(batch):
        images, cap_idx, captions = zip(*batch)
        
        pixel_values = processor(images=list(images), return_tensors="pt").pixel_values.squeeze()

        '''

        tokenized = tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=200,
            return_tensors="pt"
        )

        cap_idxs = tokenized["input_ids"]
        '''
        cap_idxs = torch.stack(cap_idx)
        return {
            "pixel_values": pixel_values,
            "captions": captions,
            "cap_idxs": cap_idxs,
        }

    splits = get_train_val_test_annotations_split()
    #train_dataset = FoodDatasetWord(splits["train"])
    #val_dataset = FoodDatasetWord(splits["val"])
    train_dataset = FoodDatasetWordLevel(splits["train"])
    val_dataset = FoodDatasetWordLevel(splits["val"])
    test_dataset = FoodDatasetWordLevel(splits["test"])

    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=True, 
        collate_fn=custom_collate_fn, 
        num_workers=training_config["num_workers"], 
        pin_memory=True, 
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=training_config["num_workers"], 
        pin_memory=True, 
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=training_config['batch_size'], 
        shuffle=False, 
        collate_fn=custom_collate_fn, 
        num_workers=training_config["num_workers"], 
        pin_memory=True, 
        persistent_workers=True
    )
    
    return train_loader, val_loader, test_loader

def setup_training(model, projection_layer, training_config):
    # Combine parameters from both model and projection layer
    params = list(model.parameters()) + list(projection_layer.parameters())
    
    optimizer = torch.optim.AdamW(
        params,  # Includes both model and projection parameters
        lr=training_config['lr'],  
        weight_decay=training_config['weight_decay']
    )
    num_epochs = training_config['epochs']
    return optimizer, num_epochs

def train_model(model, train_loader, val_loader, optimizer, num_epochs, vit_model, projection_layer, experiment_name, embedding_layer):
    # Initialize wandb
    wandb.login(key=os.getenv('WANDB_GERARD'))
    wandb.init(name=experiment_name, project="C5-G7-LLMs", config=training_config)
    wandb.watch(model, log="all", log_freq=100)

    best_val_loss = float('inf')
    global_step = 0  # Track total training steps across all epochs
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, batch in pbar:
            global_step += 1
            pbar.set_description(f"Train Epoch {epoch} - Global Step {global_step}")
            optimizer.zero_grad()
            
            # Process images
            pixel_values = batch["pixel_values"].to(device)
            with torch.no_grad():
                image_features = vit_model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            
            # Prepare inputs
            visual_embeds = projection_layer(image_features)
            labels = batch["cap_idxs"].to(device)
            #text_embeds = model.get_input_embeddings()(labels).to(device)
            text_embeds = embedding_layer(labels).to(device)
            
            # Combine embeddings
            inputs_embeds = torch.cat([
                visual_embeds.unsqueeze(1),
                text_embeds[:, :-1, :]
            ], dim=1)
            
            # Create attention mask
            attention_mask = (labels != WORD2IDX["<PAD>"]).float().to(device)
            #attention_mask = (labels != tokenizer.pad_token_id).float().to(device)
            
            # Forward pass
            outputs = model(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask
            )
            
            # Backward pass
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Log training progress
            wandb.log({
                "train_loss": loss.item(),
                "step": global_step,
                "epoch": epoch + (batch_idx / len(train_loader))  # Fractional epoch
            })

            # Perform validation every 100 steps
            if global_step % 500 == 0:
                print("Validating...")
                val_metrics = validate_model(model, val_loader, vit_model, projection_layer, embedding_layer)
                print("Done!")
                wandb.log({
                    **val_metrics,  # Unpacks all metrics
                    "step": global_step,
                    "epoch": epoch + (batch_idx / len(train_loader))
                })
                
                # Save checkpoint if it's the best so far (now using val_loss from metrics)
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    torch.save({
                        'model_state_dict': model.state_dict(), 
                        'projection_state_dict': projection_layer.state_dict(),
                        'embedding_state_dict': embedding_layer.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'global_step': global_step,
                        'loss': best_val_loss,
                    }, f'/ghome/c5mcv07/C5_G7_MCV/Week_4/LLMs/checkpoints/{experiment_name}_best_model.pth')

        # End-of-epoch validation
        avg_train_loss = train_loss / len(train_loader)
        val_metrics = validate_model(model, val_loader, vit_model, projection_layer, embedding_layer)

        # Log metrics
        wandb.log({
            "epoch": epoch+1,
            "avg_train_loss": avg_train_loss,
            **val_metrics,  # Unpacks all validation metrics
            "step": global_step
        })
        
        # Save best model (if not already saved during mid-epoch validation)
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                'model_state_dict': model.state_dict(),
                'projection_state_dict': projection_layer.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_val_loss,
                'global_step': global_step
            }, f'/ghome/c5mcv07/C5_G7_MCV/Week_4/LLMs/checkpoints/{experiment_name}_best_model.pth')

    return model, vit_model, projection_layer, embedding_layer, optimizer

def validate_model(model, val_loader, vit_model, projection_layer, embedding_layer):
    model.eval()
    val_loss = 0.0
    all_predictions = []
    all_references = []
    
    with torch.no_grad():
        n = 0
        for batch in val_loader:
            n += 1
            print(f"Validating batch {n}/{len(val_loader)}")
            # Process images
            pixel_values = batch["pixel_values"].to(device)
            image_features = vit_model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            visual_embeds = projection_layer(image_features)
            
            # Prepare inputs
            labels = batch["cap_idxs"].to(device)
            #text_embeds = model.get_input_embeddings()(labels).to(device)
            text_embeds = embedding_layer(labels)

            # Combine embeddings
            inputs_embeds = torch.cat([
                visual_embeds.unsqueeze(1),
                text_embeds[:, :-1, :]
            ], dim=1)
            
            # Create attention mask
            #attention_mask = (labels != tokenizer.pad_token_id).float().to(device)
            attention_mask = (labels != WORD2IDX["<PAD>"]).float().to(device)
            
            # Forward pass
            outputs = model(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask
            )
            
            val_loss += outputs.loss.item()
            
            # Get predictions by taking argmax of logits
            pred_ids = torch.argmax(outputs.logits, dim=-1)
                        # Initialize list for predictions
            print(f"Shape of pred_ids: {pred_ids.shape}")

            # Decode each prediction (token IDs) for the batch
            for i, pred in enumerate(pred_ids):  # Loop over batch dimension (i is index for each item in the batch)
                decoded_tokens = []
                
                # Decode each token ID in the prediction
                for token_id in pred:
                    word = IDX2WORD.get(token_id.item(), "<UNK>")  # Use <UNK> if token_id is not found
                    if word == "<EOS>" or word == "<PAD>":  # Stop at EOS or PAD tokens
                        break
                    decoded_tokens.append(word)
                
                # Join tokens to form a sentence for this particular example
                decoded_sentence = " ".join(decoded_tokens)
                all_predictions.append(decoded_sentence)  # Add the decoded sentence to all_predictions

            # Check lengths after decoding
            print(f"Length of all_predictions: {len(all_predictions)}")  # This should be 4

            references = batch["captions"]
            print(f"Length of references: {len(references)}")  # This should also be 4
 
            # Wrap references for metrics and add them to all_references
            all_references.extend([[ref] for ref in references])  # Wrap each reference in a list
    
    # Calculate metrics
    metrics = calculate_metrics(all_predictions, all_references)
    metrics["val_loss"] = val_loss / len(val_loader)
    
    return metrics

def evaluate_model(model, test_loader, vit_model, projection_layer, embedding_layer):
    model.eval()
    vit_model.eval()
    eval_loss = 0.0
    all_predictions = []
    all_references = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            print(f"Evaluating batch {batch_idx + 1}/{len(test_loader)}")

            # Process images
            pixel_values = batch["pixel_values"].to(device)
            image_features = vit_model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
            visual_embeds = projection_layer(image_features)

            # Prepare inputs
            labels = batch["cap_idxs"].to(device)
            text_embeds = embedding_layer(labels).to(device)

            # Combine embeddings
            inputs_embeds = torch.cat([
                visual_embeds.unsqueeze(1),
                text_embeds[:, :-1, :]
            ], dim=1)

            # Create attention mask
            attention_mask = (labels != WORD2IDX["<PAD>"]).float().to(device)

            # Forward pass
            outputs = model(
                inputs_embeds=inputs_embeds,
                labels=labels,
                attention_mask=attention_mask
            )

            eval_loss += outputs.loss.item()

            # Predictions
            pred_ids = torch.argmax(outputs.logits, dim=-1)

            for i, pred in enumerate(pred_ids):
                decoded_tokens = []
                for token_id in pred:
                    word = IDX2WORD.get(token_id.item(), "<UNK>")
                    if word == "<EOS>" or word == "<PAD>":
                        break
                    decoded_tokens.append(word)
                decoded_sentence = " ".join(decoded_tokens)
                all_predictions.append(decoded_sentence)

            references = batch["captions"]
            all_references.extend([[ref] for ref in references])

    # Compute metrics
    metrics = calculate_metrics(all_predictions, all_references)
    metrics["eval_loss"] = eval_loss / len(test_loader)

    # Print results
    print("\n===== Evaluation Metrics =====")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("================================\n")

    return metrics


if __name__ == "__main__":
    # Setup
    llama_model_name = "meta-llama/Llama-3.2-3B"
    training_config = {
        "lr": 1e-4,
        "batch_size": 4,
        "epochs": 5,
        "weight_decay": 1e-3,
        "num_workers": 1
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"{timestamp}_{llama_model_name.replace("/", "_")}"
    
    # Initialize models and data loaders
    processor, vit_model, peft_model, projection_layer, embedding_layer = initialize_models(llama_model_name)
    train_loader, val_loader, test_loader = create_data_loaders(processor,training_config)
    optimizer, num_epochs = setup_training(peft_model, projection_layer, training_config)
    
    # Train the model
    model, vit_model, proj_layer, emb_layer, optimizer  = train_model(
        model=peft_model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        num_epochs=num_epochs,
        vit_model=vit_model,
        projection_layer=projection_layer,
        experiment_name=experiment_name,
        embedding_layer=embedding_layer
    )

    evaluate_model(model, test_loader, vit_model, projection_layer, embedding_layer)