#!/usr/bin/env python3
import argparse
import sqlite3
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from identifur import models
from identifur.data import load_tags


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("checkpoint")
    argparser.add_argument("sample")
    argparser.add_argument("--base-model", default="vit_l_16")
    argparser.add_argument("--dataset-path", default="dataset")
    argparser.add_argument("--tag-min-post-count", default=2500, type=int)
    args = argparser.parse_args()

    db = sqlite3.connect(args.data_db)
    tags = load_tags(db, args.tag_min_post_count)
    device = torch.device("cuda")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    (model, input_size) = models.MODELS[args.base_model]
    model = model(num_classes=len(tags), pretrained=False, requires_grad=False).to(
        device
    )
    model.load_state_dict(checkpoint["model"])
    model.eval()

    outputs = model(
        transforms.ToTensor()(
            transforms.Resize(input_size)(Image.open(args.sample).convert("RGB"))
        )
        .unsqueeze(0)
        .to(device),
    )
    outputs = torch.sigmoid(outputs)
    outputs = outputs.detach().cpu()
    sorted_indices = list(reversed(np.argsort(outputs[0])))
    for i in sorted_indices[:10]:
        print(tags[i], outputs[0][i].item())


if __name__ == "__main__":
    main()
