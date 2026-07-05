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
        label = DX_TO_IDX[row['dx']]
        return (img, label)

    def _resolve_image_path(self, image_id):
        return os.path.join(self.image_dir_path, image_id + '.jpg')


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

    return weights

# -----------------------------------------------------------------------
# Transform pipelines
# -----------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def get_eval_transforms(image_size=224):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
