import random

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from constants import ANNOTATIONS_PATH, OUTPUTS_PATH


def get_annotations():
    annotations_df = pd.read_csv(ANNOTATIONS_PATH, index_col=0)
    annotations_df['Image_Name'] = annotations_df['Image_Name'].apply(lambda x: f'{x}.jpg')
    return annotations_df

def get_train_val_test_annotations_split(splits=[0.8,0.1,0.1]):
    assert sum(splits) == 1
    annotations_df = get_annotations()

    n_samples = len(annotations_df)
    train_size = int(splits[0] * n_samples)
    val_size = int(splits[1] * n_samples)

    shuffled_indices = np.random.permutation(n_samples)
    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size + val_size]
    test_indices = shuffled_indices[train_size + val_size:]

    return {
        'train':annotations_df.iloc[train_indices].reset_index(drop=True),
        'val': annotations_df.iloc[val_indices].reset_index(drop=True),
        'test': annotations_df.iloc[test_indices].reset_index(drop=True)
    }

def visualize_samples(dataset, num_samples=5):
    indices = random.sample(range(len(dataset)), num_samples)
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 5))
    
    if num_samples == 1:
        axes = [axes]  # Ensure axes is iterable
    
    for ax, idx in zip(axes, indices):
        img, caption, _ = dataset[idx]  # Get raw image and caption

        # Convert tensor image to numpy format for visualization
        img_np = img.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C)
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize for display

        ax.imshow(img_np)
        ax.set_title(caption, fontsize=10)
        ax.axis("off")
        
    save_path = OUTPUTS_PATH / "visualized_samples.png"
    plt.savefig(save_path)
    plt.close()
