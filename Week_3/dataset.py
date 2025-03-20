import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
from utils import get_train_val_test_annotations_split, visualize_samples
from constants import CHAR2IDX, CHARS, IMAGES_PATH, TEXT_MAX_LEN


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
        cap_list = list(caption)  # Convert caption to a list of characters
        final_list = [CHARS[0]]  # Add <SOS> token at the start
        final_list.extend(cap_list)  # Add the caption characters
        final_list.extend([CHARS[1]])  # Add <EOS> token at the end
        
        # Pad the caption to max_len
        gap = self.max_len - len(final_list)
        final_list.extend([CHARS[2]] * gap)  # Add <PAD> tokens as needed

        # Convert characters to indices using char2idx
        print(CHAR2IDX)
        print(final_list)
        cap_idx = [CHAR2IDX[i] for i in final_list]
        
        return img, caption, cap_idx


if __name__ == "__main__":
    splits = get_train_val_test_annotations_split()
    train_annotations = splits["train"]  # Visualize training examples
    dataset = FoodDataset(train_annotations)
    visualize_samples(dataset, num_samples=3)
