import torch
import torch.nn as nn
from torchvision import models

class AdvancedTransferModel(nn.Module):
    def __init__(self, num_classes=8, freeze_features=False):
        super(AdvancedTransferModel, self).__init__()
        
        # Load the massive ResNet50
        # More layers
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # ResNet50 outputs 2048 flat features (instead of ResNet18's 512)
        num_flat_features = self.resnet.fc.in_features 
        
        # Add Dropout before the final layer to aggressively prevent overfitting
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_flat_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)