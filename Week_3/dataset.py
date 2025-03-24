import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from utils import get_train_val_test_annotations_split, visualize_samples
from constants import CHAR2IDX, CHARS, IMAGES_PATH, TEXT_MAX_LEN, WORD2IDX, WORD_MAX_LEN, SUBWORD_MAX_LEN, tokenizer
from unidecode import unidecode
import numpy as np
import re



class FoodDataset(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations
        self.max_len = TEXT_MAX_LEN
        self.tsfms = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]

        ## Load image
        img_name = item.Image_Name  # Single image name per dish
        img = Image.open(str(IMAGES_PATH / img_name)).convert('RGB')
        img = self.tsfms(img)
    
        ## Caption processing
        caption = item.Title  # Single caption for the image
        clean_caption = re.sub(r'[^a-zA-Z0-9 ]', '', caption)
        cap_list = list(clean_caption)  # Convert caption to a list of characters
        final_list = [CHARS[0]]  # Add <SOS> token at the start
        final_list.extend([char for char in cap_list if char != '']) # Add the caption characters
        final_list.extend([CHARS[1]])  # Add <EOS> token at the end
        
        # Pad the caption to max_len
        gap = self.max_len - len(final_list)
        final_list.extend([CHARS[2]] * gap)  # Add <PAD> tokens as needed

        # Convert characters to indices using char2idx
        final_list = [unidecode(char) for char in final_list]
        cap_idx = np.array([CHAR2IDX[i] for i in final_list], dtype=np.int64)  # Convert to numpy array
        cap_idx = torch.tensor(cap_idx)
        return img, clean_caption, cap_idx
    
class FoodDatasetWord(Dataset):
    def __init__(self, annotations):
        self.annotations = annotations
        self.max_len = WORD_MAX_LEN
        self.tsfms = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]

        ## Load image
        img_name = item.Image_Name  # Single image name per dish
        img = Image.open(str(IMAGES_PATH / img_name)).convert('RGB')
        img = self.tsfms(img)
    
        ## Caption processing
        caption = item.Title  # Single caption for the image
        clean_caption = re.sub(r'[^a-zA-Z0-9 ]', '', caption)
        words = clean_caption.split()  # Convert caption to a list of characters
        final_list = [WORD2IDX['<SOS>']]   # Add <SOS> token at the start
        for word in words:
            word = unidecode(word)  # Remove accents from characters if needed
            # Add the word index to the list; use <UNK> for words that are not in the vocab
            final_list.append(WORD2IDX.get(word, WORD2IDX['<UNK>']))
        final_list.append(WORD2IDX['<EOS>'])  # Add <EOS> token at the end# Add <EOS> token at the end
        
        # Pad the caption to max_len
        gap = self.max_len - len(final_list)
        final_list.extend([WORD2IDX['<PAD>']] * gap)  # Add <PAD> tokens as needed

        cap_idx = np.array(final_list, dtype=np.int64)  # Convert to numpy array
        cap_idx = torch.tensor(cap_idx)
        return img, clean_caption, cap_idx

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
        self.tokenizer = tokenizer  # Use WordPiece tokenizer for tokenization
        self.tsfms = torch.nn.Sequential(
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize((224, 224), antialias=True),
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        )

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations.iloc[idx]

        ## Load image
        img_name = item.Image_Name  # Single image name per dish
        img = Image.open(str(IMAGES_PATH / img_name)).convert('RGB')
        img = self.tsfms(img)
    
        ## Caption processing
        caption = item.Title  # Single caption for the image

        # Clean up caption text (if needed)
        clean_caption = re.sub(r'[^a-zA-Z0-9 ]', '', caption)

        # Tokenize caption using WordPiece tokenizer
        encoded = self.tokenizer.encode(clean_caption)

        # Add <SOS> and <EOS> tokens (based on the tokenizer)
        final_list = [self.tokenizer.token_to_id('[CLS]')]  # Using tokenizer's [CLS] as <SOS>
        final_list.extend(encoded.ids)  # Add tokenized words
        final_list.append(self.tokenizer.token_to_id('[SEP]'))  # Using tokenizer's [SEP] as <EOS>

        # Pad the caption to max_len
        gap = self.max_len - len(final_list)
        if gap > 0:
            final_list.extend([self.tokenizer.token_to_id('[PAD]')]* gap)  # Add padding tokens if needed

        cap_idx = np.array(final_list, dtype=np.int64)  # Convert to numpy array
        cap_idx = torch.tensor(cap_idx)  # Convert to torch tensor

        return img, clean_caption, cap_idx

if __name__ == "__main__":
    splits = get_train_val_test_annotations_split()
    train_annotations = splits["train"]  # Visualize training examples
    dataset = FoodDataset(train_annotations)
    visualize_samples(dataset, num_samples=3)
