from collections import OrderedDict
from torchvision import models
import torch
from torch import optim, nn
from torchmetrics import Accuracy
import pytorch_lightning as pl


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


MODELS = {
    "resnet152": (resnet152, (224, 224)),
    "vit_l_16": (vit_l_16, (224, 224)),
}


class LitModel(pl.LightningModule):
    def __init__(self, model, num_labels, lr=2e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task="multilabel", num_labels=num_labels)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def training_step(self, batch):
        y_pred, y = batch
        y_pred = self.forward(y_pred)

        loss = self.criterion(y_pred, y)
        self.log("train/loss", loss)

        acc = self.accuracy(y_pred, y)
        self.log("train/acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        y_pred, y = batch
        y_pred = self.forward(y_pred)

        loss = self.criterion(y_pred, y)
        self.log("val/loss", loss)

        acc = self.accuracy(y_pred, y)
        self.log("val/acc", acc)

        return loss

    def test_step(self, batch, batch_idx):
        y_pred, y = batch
        y_pred = self.forward(y_pred)
        loss = self.criterion(y_pred, y)

        return {"loss": loss, "y_pred": y_pred, "y": y}

    def test_epoch_end(self, outputs):
        loss = torch.stack([output["loss"] for output in outputs]).mean()  # type: ignore
        self.log("test/loss", loss)

        y_pred = torch.cat([output["y_pred"] for output in outputs], dim=0)  # type: ignore
        ys = torch.cat([output["y"] for output in outputs], dim=0)  # type: ignore
        acc = self.accuracy(y_pred, ys)
        self.log("test/acc", acc)

        self.test_ys = ys
        self.test_y_pred = y_pred
