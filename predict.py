#!/usr/bin/env python3
import numpy as np
import argparse
from PIL import Image
import torch
from torchvision import transforms
from identifur import models


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("checkpoint")
    argparser.add_argument("sample")
    argparser.add_argument("--base-model", default="vit_l_16")
    argparser.add_argument("--labels-path", default="labels")
    args = argparser.parse_args()

    device = torch.device("cuda")

    with open(args.labels_path, "rt", encoding="utf-8") as f:
        labels = [line.rstrip("\n") for line in f]

    (model, input_size) = models.MODELS[args.base_model]
    model = models.LitModel.load_from_checkpoint(
        args.checkpoint,
        model=model,
        weights=None,
        labels=labels,
        requires_grad=False,
    ).to(device)
    model.eval()
    model.freeze()

    y_pred = model(
        transforms.ToTensor()(
            transforms.Resize(input_size)(Image.open(args.sample).convert("RGB"))
        )
        .unsqueeze(0)
        .to(device),
    )
    y_pred = torch.sigmoid(y_pred)
    y_pred = y_pred.detach().cpu()
    y_pred = y_pred[0]

    sorted_indices = list(reversed(np.argsort(y_pred)))
    for i in sorted_indices[:10]:
        print(labels[i], y_pred[i].item())


if __name__ == "__main__":
    main()
