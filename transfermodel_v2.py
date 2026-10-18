import torch
import torch.nn as nn
from torchvision import models

class AdvancedTransferModel(nn.Module):
    def __init__(self, num_classes=8, freeze_features=True):
        super(AdvancedTransferModel, self).__init__()
        
        # 1. Download the massive pre-trained ResNet50
        # IMAGENET1K_V2 loads the most advanced pre-calculated math weights
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 2. Freeze the Feature Extractors
        # We tell PyTorch NOT to calculate gradients for the base layers to protect the math.
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # 3. The Brain Transplant (Swapping the Linear Layer)
        # ResNet50 is much thicker than ResNet18. It passes 2048 flat features to its final layer!
        num_flat_features = self.resnet.fc.in_features 
        
        # Replace the original layer with a Dropout (to prevent overfitting) and our new 8-class layer
        self.resnet.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_flat_features, num_classes)
        )

    def forward(self, x):
        # Pass the image directly through the modified ResNet50
        return self.resnet(x)

# --- Test the Architecture ---
if __name__ == "__main__":
    # Initialize the new model
    model = AdvancedTransferModel(num_classes=8, freeze_features=True)
    
    # Pass a fake image through it (ResNet handles 120x120 perfectly!)
    dummy_image = torch.randn(1, 3, 120, 120)
    output = model(dummy_image)
    
    print("Pre-trained ResNet50 successfully loaded and modified!")
    print("Output Shape:", output.shape) # Should be [1, 8]