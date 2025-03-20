import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Model
from dataset import FoodDataset
from utils import get_train_val_test_annotations_split

def train_and_evaluate():
    # Get train, validation, and test splits
    splits = get_train_val_test_annotations_split()
    train_annotations = splits["train"]
    val_annotations = splits["val"]
    test_annotations = splits["test"]
    
    # Create dataset and dataloader
    train_dataset = FoodDataset(train_annotations)
    val_dataset = FoodDataset(val_annotations)
    test_dataset = FoodDataset(test_annotations)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Model initialization
    model = Model()
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
        for img, caption, cap_idx in train_loader:
            print(cap_idx)
            img, cap_idx = img.to(device), torch.tensor(cap_idx).to(device)
            
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, cap_idx)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
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
    with torch.no_grad():
        for img, caption, cap_idx in test_loader:
            img, cap_idx = img.to(device), torch.tensor(cap_idx).to(device)
            outputs = model(img)
            loss = criterion(outputs, cap_idx)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")
    
    return model

# Run training and evaluation
trained_model = train_and_evaluate()
