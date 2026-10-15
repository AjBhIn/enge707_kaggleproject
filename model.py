# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=8):
        super(SimpleCNN, self).__init__()
        
        # --- 1. The Feature Extractors ---
        
        # Takes in 3 color channels (RGB), outputs 16 feature maps
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks image 224 -> 112
        
        # Takes in 16 feature maps, outputs 32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks image 112 -> 56
        
        # Takes in 32 feature maps, outputs 64
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrinks image 56 -> 28
        
        # --- 2. The Decision Maker ---
        # these are the fully connected layers
        # Flattened size: 64 feature maps * 28 height * 28 width
        self.fc1 = nn.Linear(64 * 28 * 28, 512)
        
        # Final output layer giving us 8 scores (one for each class)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout randomly turns off 50% of neurons during training so the 
        # model doesn't over-rely on just a few pixels (prevents overfitting)
        self.dropout = nn.Dropout(0.5)

    
    def forward(self, x):
        """This defines the exact path the image takes through the network."""
        # Block 1
        x = self.pool1(F.relu(self.conv1(x)))
        # Block 2
        x = self.pool2(F.relu(self.conv2(x)))
        # Block 3
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten the 3D grid into a 1D list
        x = torch.flatten(x, 1) 
        
        # Decision block
        # these are the fully connected layers
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x) # Output the final 8 raw scores
        
        return x