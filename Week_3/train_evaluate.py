import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Model, ModelWords, ModelWordsPiece
from dataset import FoodDataset, FoodDatasetWord, FoodDatasetSubWord
from utils import get_train_val_test_annotations_split
from tqdm import tqdm
from constants import tokenizer, WORD2IDX, IDX2WORD
import evaluate


def calculate_metrics(predictions, references):

    # Load the metrics
    bleu = evaluate.load("bleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    # BLEU-1 and BLEU-2 
    bleu1 = bleu.compute(predictions=predictions, references=references, max_order=1)['bleu']
    bleu2 = bleu.compute(predictions=predictions, references=references, max_order=2)['bleu']
    
    # ROUGE-L 
    rouge = rouge.compute(predictions=predictions, references=references)['rougeL']
    
    # METEOR 
    meteor = meteor.compute(predictions=predictions, references=references)['meteor']
    
    return bleu1, bleu2, rouge, meteor

def train_and_evaluate():
    # Get train, validation, and test splits
    splits = get_train_val_test_annotations_split()
    train_annotations = splits["train"]
    val_annotations = splits["val"]
    test_annotations = splits["test"]
    
    # Create dataset and dataloader
    train_dataset = FoodDatasetWord(train_annotations)
    val_dataset = FoodDatasetWord(val_annotations)
    test_dataset = FoodDatasetWord(test_annotations)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Model initialization
    model = ModelWords()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        t= tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch', leave= False)
        for img, caption, cap_idx in t:
            #print(cap_idx.shape, cap_idx)
            img, cap_idx = img.to(device), cap_idx.to(device) 
            optimizer.zero_grad()
            outputs = model(img)
            #print(outputs.shape)
            #print(cap_idx.shape)
            loss = criterion(outputs, cap_idx)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            t.set_postfix(loss=loss.item())
                
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
            
        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for img, caption, cap_idx in val_loader:
                img, cap_idx = img.to(device), torch.tensor(cap_idx).to(device)
                outputs = model(img)
                loss = criterion(outputs, cap_idx)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Evaluation on test set
    model.eval()
    test_loss = 0.0
    predictions = []
    references = []
    with torch.no_grad():
        for img, caption, cap_idx in test_loader:
            img, cap_idx = img.to(device), torch.tensor(cap_idx).to(device)
            outputs = model(img)
            loss = criterion(outputs, cap_idx)
            # Convert the predicted output back to tokens (e.g., subword tokens)
            pred_caption = torch.argmax(outputs, dim=2)  # shape (batch_size, seq_len)
            pred_caption = pred_caption.cpu().numpy()
            pred_texts = []
            i = 0
            for pred in pred_caption:
                print(i)
                pred_text = " ".join([IDX2WORD[idx] for idx in pred if idx != WORD2IDX['<PAD>']])  # Skip padding index
                pred_texts.append(pred_text)
                i += 1

            # Get the reference captions from the dataset
            ref_texts = [caption for caption in caption]
            
            # Store predictions and references
            predictions.extend(pred_texts)
            references.extend([[ref] for ref in ref_texts])  # wrap each reference in a list for BLEU
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    # Calculate metrics
    print(len(predictions), len(references))
    bleu1, bleu2, rouge_l, meteor = calculate_metrics(predictions, references)
    print(f"BLEU-1: {bleu1:.4f}, BLEU-2: {bleu2:.4f}, ROUGE-L: {rouge_l:.4f}, METEOR: {meteor:.4f}")
    print(predictions[:10], references[:10])
    
    
    return model

# Run training and evaluation
trained_model = train_and_evaluate()
