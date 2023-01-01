from collections import OrderedDict
from torchvision import models
import torch
from torch import optim, nn
from torchmetrics.classification.accuracy import MultilabelAccuracy
import pytorch_lightning as pl
import itertools
import timm


def _make_resnet_model(f, layers_to_freeze=6):
    def _model(pretrained, num_labels, requires_grad=False):
        model = f(weights="DEFAULT" if pretrained else None)

        for child in itertools.islice(model.children(), layers_to_freeze):
            for param in child.parameters():
                param.requires_grad = requires_grad

        model.fc = nn.Linear(model.fc.in_features, num_labels)
        return model

    return _model


def _make_vit_model(f):
    def _model(pretrained, num_labels, requires_grad=False):
        model = f(weights="DEFAULT" if pretrained else None)
        model.heads = nn.Sequential(
            OrderedDict(
                [
                    ("head", nn.Linear(model.hidden_dim, num_labels)),
                ]
            )
        )
        return model

    return _model


def _make_convnext_model(f, layers_to_freeze=25):
    def _model(pretrained, num_labels, requires_grad=False):
        model: models.ConvNeXt = f(weights="DEFAULT" if pretrained else None)

        for child in itertools.islice(
            (layer for block in model.features for layer in block),
            layers_to_freeze,
        ):
            for param in child.parameters():
                param.requires_grad = requires_grad

        fc: nn.Linear = model.classifier[-1]  # type: ignore
        model.classifier[-1] = nn.Linear(fc.in_features, num_labels)
        return model

    return _model


def _make_efficientnet_v2_model(f, layers_to_freeze=6):
    def _model(pretrained, num_labels, requires_grad=False):
        model: models.EfficientNet = f(weights="DEFAULT" if pretrained else None)

        for layer in itertools.islice(model.features, layers_to_freeze):
            for param in layer.parameters():
                param.requires_grad = requires_grad

        fc: nn.Linear = model.classifier[-1]  # type: ignore
        model.classifier[-1] = nn.Linear(fc.in_features, num_labels)
        return model

    return _model


def _make_deit_model(name, layers_to_freeze=10):
    def _model(pretrained, num_labels, requires_grad=False):
        model: timm.models.VisionTransformer = torch.hub.load(
            "facebookresearch/deit:main", name, pretrained=pretrained
        )

        for layer in itertools.islice(model.blocks, layers_to_freeze):
            for param in layer.parameters():
                param.requires_grad = requires_grad

        model.head = nn.Linear(model.head.in_features, num_labels)
        return model

    return _model


MODELS = {
    "resnet152": (_make_resnet_model(models.resnet152), (224, 224)),
    "convnext_large": (_make_convnext_model(models.convnext_large), (224, 224)),
    "convnext_base": (_make_convnext_model(models.convnext_base), (224, 224)),
    "efficientnet_v2_l": (
        _make_efficientnet_v2_model(models.efficientnet_v2_l),
        (224, 224),
    ),
    "vit_b_16": (_make_vit_model(models.vit_b_16), (224, 224)),
    "vit_b_32": (_make_vit_model(models.vit_b_32), (224, 224)),
    "vit_l_16": (_make_vit_model(models.vit_l_16), (224, 224)),
    "vit_l_32": (_make_vit_model(models.vit_l_32), (224, 224)),
    "deit_base_distilled_patch16_384": (
        _make_deit_model("deit_base_distilled_patch16_384"),
        (384, 384),
    ),
}


class LitModel(pl.LightningModule):
    def __init__(
        self, model, num_labels, lr=1e-4, pretrained=True, requires_grad=False
    ):
        super().__init__()
        self.model = model(pretrained, num_labels, requires_grad)
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = MultilabelAccuracy(num_labels=num_labels, threshold=0.8)

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
