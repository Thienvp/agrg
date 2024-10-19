import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
import os
import torch
from torchvision import transforms

class ChestXRayDataset(Dataset):
    """
    Custom dataset class for loading chest X-ray images and their corresponding labels.
    Args:
        df (pd.DataFrame): DataFrame containing image paths and labels.
        root_dir (str): Directory containing the images.
        processor (ViTImageProcessor, optional): Processor for preprocessing images.
        train (bool): Flag to indicate if the dataset is for training.
    """
    def __init__(self, df, root_dir, processor=None, train=True):
        self.data = df
        self.root_dir = root_dir
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_name = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        try:
            image = Image.open(img_name + ".jpg").convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError) as e:
            print(f"Warning: Could not load image {img_name}. Error: {e}")
            return None, None

        labels = self.data.iloc[idx, 2:]
        labels = torch.tensor(labels, dtype=torch.float32)

        # Apply the processor if available
        if self.processor:
            image = self.processor(image, return_tensors="pt")
            image = image.pixel_values[0]

        return image, labels