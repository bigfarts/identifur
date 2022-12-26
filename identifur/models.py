from collections import OrderedDict
from torchvision import models
from torch import nn


def set_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def resnet152(pretrained, num_classes, requires_grad):
    model = models.resnet152(
        weights=models.ResNet152_Weights.DEFAULT if pretrained else None
    )
    set_requires_grad(model, requires_grad)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def vit_l_16(pretrained, num_classes, requires_grad):
    model = models.vit_l_16(
        weights=models.ViT_L_16_Weights.DEFAULT if pretrained else None,
    )
    set_requires_grad(model, requires_grad)

    model.heads = nn.Sequential(
        OrderedDict(
            [
                ("head", nn.Linear(model.hidden_dim, num_classes)),
            ]
        )
    )
    return model
