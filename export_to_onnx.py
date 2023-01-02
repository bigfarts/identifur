#!/usr/bin/env python3
import argparse
import torch
from identifur import models


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("checkpoint")
    argparser.add_argument("--device", default="cuda")
    argparser.add_argument("--base-model", default="vit_l_16")
    argparser.add_argument("--tags-path", default="tags")
    argparser.add_argument("--output-path", default="identifur.onnx.pb")
    args = argparser.parse_args()

    device = torch.device(args.device)

    with open(args.tags_path, "rt", encoding="utf-8") as f:
        num_labels = sum(1 for _ in f) + 3

    (model, input_size) = models.MODELS[args.base_model]
    model = models.LitModel.load_from_checkpoint(
        args.checkpoint,
        model=model,
        pretrained=False,
        num_labels=num_labels,
        requires_grad=False,
    ).to(device)
    model.eval()
    model.freeze()

    torch.onnx.export(
        model,
        torch.zeros((3, *input_size)).unsqueeze(0),
        args.output_path,
        verbose=True,
        input_names=["images"],
        output_names=["labels"],
    )


if __name__ == "__main__":
    main()
