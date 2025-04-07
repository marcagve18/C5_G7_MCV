import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from .utils import get_train_val_test_annotations_split, visualize_samples
from .constants import CHAR2IDX, CHARS, IMAGES_PATH, TEXT_MAX_LEN, WORD2IDX, WORD_MAX_LEN, SUBWORD_MAX_LEN, tokenizer
from unidecode import unidecode
import numpy as np
import re

class FoodDataset(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations
        self.max_len = TEXT_MAX_LEN

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]

        ## Load image
        img_name = item.Image_Name  # Single image name per dish
        img = Image.open(str(IMAGES_PATH / img_name)).convert('RGB')
    
        ## Caption processing
        caption = item.Title  # Single caption for the image
        clean_caption = re.sub(r'[^a-zA-Z0-9 ]', '', caption)
 
        return img, clean_caption
    
class FoodDatasetWord(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations
        self.max_len = WORD_MAX_LEN

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]

        ## Load image
        img_name = item.Image_Name  # Single image name per dish
        img = Image.open(str(IMAGES_PATH / img_name)).convert('RGB')
    
        ## Caption processing
        caption = item.Title  # Single caption for the image
        clean_caption = re.sub(r'[^a-zA-Z0-9 ]', '', caption)
        
        return img, clean_caption

class FoodDatasetSubWord(Dataset):
    def __init__(self, annotations):
        """
        Parameters:
            annotations (DataFrame): The DataFrame containing the dataset annotations.
            tokenizer (Tokenizer): The tokenized object (WordPiece or other).
            max_len (int): The maximum length for tokenized captions.
        """
        self.annotations = annotations
        self.max_len = SUBWORD_MAX_LEN

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]

        ## Load image
        img_name = item.Image_Name  # Single image name per dish
        img = Image.open(str(IMAGES_PATH / img_name)).convert('RGB')
    
        ## Caption processing
        caption = item.Title  # Single caption for the image

        # Clean up caption text (if needed)
        clean_caption = re.sub(r'[^a-zA-Z0-9 ]', '', caption)

        return img, clean_caption
    
class FoodDatasetWordLevel(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations
        self.word2idx = WORD2IDX
        self.max_len = WORD_MAX_LEN

    def tokenize(self, caption):
        # Clean and tokenize
        caption = caption.replace("-", " ")
        caption = re.sub(r'[^a-zA-Z0-9 ]', '', caption).lower()
        words = caption.split()

        tokens = ["<SOS>"] + [unidecode(word) for word in words] + ["<EOS>"]
        token_ids = [self.word2idx.get(word, self.word2idx["<UNK>"]) for word in tokens]

        # Truncate + pad
        token_ids = token_ids[:self.max_len]
        padding = [self.word2idx["<PAD>"]] * (self.max_len - len(token_ids))
        return torch.tensor(token_ids + padding)

    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]

        ## Load image
        img_name = item.Image_Name  # Single image name per dish
        img = Image.open(str(IMAGES_PATH / img_name)).convert('RGB')
    
        ## Caption processing
        caption = item.Title  # Single caption for the image

        # Get tokenized caption
        token_ids = self.tokenize(caption)
        
        # Return image, tokenized caption, and original caption
        return img, token_ids, caption

    def __len__(self):
        return len(self.annotations)


if __name__ == "__main__":
    splits = get_train_val_test_annotations_split()
    train_annotations = splits["train"]  # Visualize training examples
    dataset = FoodDataset(train_annotations)
    visualize_samples(dataset, num_samples=3)
