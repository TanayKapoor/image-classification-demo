import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class CustomResNet50(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet50, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

def get_resnet50_model(num_classes):
    return CustomResNet50(num_classes)

# Example usage:
# model = get_resnet50_model(1000)  # for ImageNet
