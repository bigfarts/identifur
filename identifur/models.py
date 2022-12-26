from torchvision import models
from torch import nn

def model(pretrained, out_features, requires_grad):
    model = models.resnet152(
        progress=True, weights=models.ResNet152_Weights.DEFAULT if pretrained else None
    )

    for param in model.parameters():
        param.requires_grad = requires_grad

    model.fc = nn.Linear(2048, out_features)
    return model
