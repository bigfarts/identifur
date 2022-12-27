#!/usr/bin/env python3
import numpy as np
import argparse
import sqlite3
from PIL import Image
import torch
from torchvision import transforms
from identifur import models
from identifur.data import load_tags
import pytorch_lightning as pl


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("checkpoint")
    argparser.add_argument("sample")
    argparser.add_argument("--base-model", default="vit_l_16")
    argparser.add_argument("--dataset-path", default="dataset")
    argparser.add_argument("--tag-min-post-count", default=2500, type=int)
    args = argparser.parse_args()

    with sqlite3.connect(f"file:{args.data_db}?mode=ro", uri=True) as db:
        tags = load_tags(db, args.tag_min_post_count)

    device = torch.device("cuda")

    (model, input_size) = models.MODELS[args.base_model]
    model = models.LitModel.load_from_checkpoint(
        args.checkpoint,
        model=model(pretrained=True, requires_grad=False, num_classes=len(tags)),
        num_labels=len(tags),
    ).to(device)
    model.eval()

    with torch.no_grad():
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
            print(tags[i], y_pred[i].item())


if __name__ == "__main__":
    main()
