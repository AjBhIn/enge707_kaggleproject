import torch
import torch.nn as nn
from torchvision import models

class TransferResNet(nn.Module):
    def __init__(self, num_classes=8, freeze_features=True):
        super(TransferResNet, self).__init__()
        
        # 1. Download the pre-trained ResNet18 (The PhD Student)
        # IMAGENET1K_V1 loads the best pre-calculated math weights
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # 2. Freeze the Feature Extractors
        # We tell PyTorch NOT to calculate gradients for the base layers. 
        # We don't want to accidentally ruin the math it already learned!
        if freeze_features:
            for param in self.resnet.parameters():
                param.requires_grad = False
                
        # 3. The Brain Transplant (Swapping the Linear Layer)
        # Find out exactly how many flat features ResNet passes to its final layer (it's 512)
        num_flat_features = self.resnet.fc.in_features 
        
        # Replace the original 1000-class layer with our new 8-class layer
        # By default, PyTorch leaves requires_grad=True for brand new layers
        self.resnet.fc = nn.Linear(num_flat_features, num_classes)

    def forward(self, x):
        # We just pass the image directly through the modified ResNet
        return self.resnet(x)

# # --- Test the Architecture ---
# if __name__ == "__main__":
#     # Initialize the new model
#     model = TransferResNet(num_classes=8)
    
#     # Pass a fake image through it
#     dummy_image = torch.randn(1, 3, 224, 224)
#     output = model(dummy_image)
    
#     print("Pre-trained ResNet18 successfully loaded and modified!")
#     print("Output Shape:", output.shape) # Should still be [1, 8]