#!/usr/bin/env python3
import sys
import numpy as np
import argparse
from PIL import Image
import torch
from torchvision import transforms
from identifur import models


def _image_without_transparency(img):
    img.load()
    img2 = Image.new("RGB", img.size, (255, 255, 255))
    img2.paste(img, mask=img.split()[3])
    return img2


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("checkpoint")
    argparser.add_argument("--device", default="cuda")
    argparser.add_argument("--base-model", default="vit_l_16")
    argparser.add_argument("--tags-path", default="tags")
    args = argparser.parse_args()

    device = torch.device(args.device)

    with open(args.tags_path, "rt", encoding="utf-8") as f:
        labels = [line.rstrip("\n") for line in f]
    labels += ["rating: s", "rating: q", "rating: e"]

    (model, input_size) = models.MODELS[args.base_model]
    model = models.LitModel.load_from_checkpoint(
        args.checkpoint,
        model=model,
        weights=None,
        num_labels=len(labels),
        requires_grad=False,
    ).to(device)
    model.eval()
    model.freeze()

    y_pred = model(
        transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(input_size),
            ]
        )(_image_without_transparency(Image.open(sys.stdin.buffer)))
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
