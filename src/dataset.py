import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

DX_TO_IDX = {
    'akiec': 0,
    'bcc': 1,
    'bkl': 2,
    'df': 3,
    'mel': 4,
    'nv': 5,
    'vasc': 6,
}
IDX_TO_DX = {v: k for k, v in DX_TO_IDX.items()}

num_classes = 7

# -----------------------------------------------------------------------
# Dataset class
# -----------------------------------------------------------------------
class HAM10000Dataset(Dataset):
    def __init__(self, metadata_df, image_dir_path, transform=None):
        """
        Args:
            metadata_df: DataFrame subset for this split (train/val/test)
            image_dir_path: directory path for location of images
            transform: torchvision transform pipeline to apply
        """
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.image_dir_path = image_dir_path
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        path = self._resolve_image_path(row['image_id'])
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = DX_TO_IDX(row['dx'])
        return (img, label)

    def _resolve_image_path(self, image_id):
        return self.image_dir_path + image_id + '.jpg'


# -----------------------------------------------------------------------
# Class weight computation
# -----------------------------------------------------------------------
def compute_class_weights(train_df):
    """
    Compute class weights inversely proportional to frequency,
    based on the IMAGE-level distribution in the training split.

    Returns:
        torch.Tensor of shape (num_classes,), ordered to match DX_TO_IDX
    """
    
    counts = train_df.value_counts('dx')
    total = counts.sum()

    weights = torch.empty(num_classes)

    for dx in DX_TO_IDX:
        weights[DX_TO_IDX[dx]] = total / (num_classes * counts[dx])
