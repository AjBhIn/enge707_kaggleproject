# data_utils.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

class FacialFeatureDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.transform = transform
        
        if 'class' in dataframe.columns:
            unique_classes = sorted(dataframe['class'].unique())
            self.class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        else:
            self.class_to_idx = None
            
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # index based look up
        img_name = self.dataframe.iloc[idx]['file_path']
        img_path = os.path.join(self.img_dir, img_name).replace('\\', '/')
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        if self.class_to_idx is not None:
            label_str = self.dataframe.iloc[idx]['class']
            label = self.class_to_idx[label_str]
            return image, label
        else:
            return image, img_name


# Image transformatio : more : to avoid overfitting
def get_transforms(is_train=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Slight random zooming
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Random lighting changes
            transforms.ToTensor(),
            normalize
        ])
    else:
        return transforms.Compose([
            # Changed the Image Size to smaller 
            # as most of the images are smaller
            # increases speed
            transforms.Resize((130, 130)),
            transforms.ToTensor(),
            normalize
        ])

def build_training_dataloaders(csv_path, img_dir, batch_size=32, val_split=0.2):
    df = pd.read_csv(csv_path)
    
    # stratify balances / gives out data with balance of each class
    train_df, val_df = train_test_split(df, test_size=val_split, 
                                        stratify=df['class'], random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    # Just initialising it; the data will be loaded when needed (improves performance)
    train_dataset = FacialFeatureDataset(train_df, img_dir, transform=get_transforms(is_train=True))
    val_dataset = FacialFeatureDataset(val_df, img_dir, transform=get_transforms(is_train=False))
    
    # Wrapping dataset into the dataloader: it takes groups of data for the model
    # shuffle=True for train is for the randomness of the data
    # further, this what runs the methods inside the FacialFeatureDataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader



# --- Make the data Ready before feeding to ready model ---

class TestDataset(Dataset):
    def __init__(self, submission_df, img_dir, transform=None):
        self.dataframe = submission_df
        self.img_dir = img_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Format the ID to match the filenames (e.g., '1' -> '00001.jpg')
        img_id = str(self.dataframe.iloc[idx]['image_id']).zfill(5)
        img_name = f"{img_id}.jpg"
        img_path = os.path.join(self.img_dir, img_name).replace('\\', '/')
        
        # Load image and ensure 3 color channels
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # Return the processed image and its original ID
        return image, self.dataframe.iloc[idx]['image_id']

def build_test_dataloader(sample_sub_path, img_dir, batch_size=32):
    """Utility function to load the Kaggle test set for inference."""
    # Load the Kaggle sample submission file
    sub_df = pd.read_csv(sample_sub_path)
    
    # We use get_transforms(is_train=False) because we NEVER augment test data
    test_dataset = TestDataset(sub_df, img_dir=img_dir, transform=get_transforms(is_train=False))
    
    # shuffle=False is critical! We must keep the exact order of the CSV for Kaggle
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Test Pipeline Ready! Images to predict: {len(sub_df)}")
    return test_loader, sub_df