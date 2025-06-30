import torch.nn as nn
from torchvision import models

def get_model():
    model = models.mobilenet_v2(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False  # freeze base
    
    model.classifier[1] = nn.Linear(model.last_channel, 1)
    return model
