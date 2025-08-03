# src/models/ai_detector_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AIGeneratedDetector(nn.Module):
    def __init__(self, num_classes=2): # 2 classes: real and AI-generated
        super().__init__()
        # Using a pre-trained ResNet-18 as a backbone, modifying the final layer
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Freeze all parameters in the backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Replace the final classification layer
        # The input features to the last linear layer of ResNet-18 is 512
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes) # Output 2 classes

    def forward(self, x):
        # Input 'x' is an image tensor (e.g., [batch_size, 3, 224, 224])
        return self.backbone(x)