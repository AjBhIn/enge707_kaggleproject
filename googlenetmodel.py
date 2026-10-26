import torch.nn as nn
from torchvision import models

class GoogLeNetTransfer(nn.Module):
    def __init__(self, num_classes=8, freeze_features=True):
        super(GoogLeNetTransfer, self).__init__()
        
        # Initialize GoogLeNet with weights 
        self.googlenet = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1, aux_logits=True)
        
        # Freeze foundational features if requested
        if freeze_features:
            for param in self.googlenet.parameters():
                param.requires_grad = False
                
        # Unfreeze and resize the MAIN classification head
        num_ftrs = self.googlenet.fc.in_features
        self.googlenet.fc = nn.Linear(num_ftrs, num_classes)
        self.googlenet.fc.requires_grad = True
        
        # Unfreeze and resize the AUXILIARY classifiers (Notice we target 'fc2' here)
        if self.googlenet.aux1 is not None:
            num_ftrs_aux1 = self.googlenet.aux1.fc2.in_features
            self.googlenet.aux1.fc2 = nn.Linear(num_ftrs_aux1, num_classes)
            self.googlenet.aux1.fc2.requires_grad = True
            
        if self.googlenet.aux2 is not None:
            num_ftrs_aux2 = self.googlenet.aux2.fc2.in_features
            self.googlenet.aux2.fc2 = nn.Linear(num_ftrs_aux2, num_classes)
            self.googlenet.aux2.fc2.requires_grad = True

    def forward(self, x):
        # In training mode, GoogLeNet returns a tuple: (outputs, aux2, aux1).
        # We extract and return only the main classification output (index 0).
        if self.training:
            output = self.googlenet(x)
            if isinstance(output, tuple) or isinstance(output, list):
                return output[0]  
            return output
        else:
            return self.googlenet(x)