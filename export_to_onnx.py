#!/usr/bin/env python3
import argparse
import torch
from identifur import models
import onnx


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("checkpoint")
    argparser.add_argument("--device", default="cuda")
    argparser.add_argument("--base-model", default="resnet152")
    argparser.add_argument("--tags-path", default="tags")
    argparser.add_argument("--output-path", default="identifur.onnx.pb")
    args = argparser.parse_args()

    device = torch.device(args.device)

    with open(args.tags_path, "rt", encoding="utf-8") as f:
        tags = [line.rstrip("\n") for line in f]

    (model, input_size) = models.MODELS[args.base_model]
    model = models.LitModel.load_from_checkpoint(
        args.checkpoint,
        model=model,
        pretrained=False,
        num_labels=len(tags) + 3,
        requires_grad=False,
    ).to(device)
    model.eval()
    model.freeze()

    torch.onnx.export(
        model,
        torch.zeros((3, *input_size)).unsqueeze(0),
        args.output_path,
        input_names=["images"],
        output_names=["labels"],
    )

    model = onnx.load(args.output_path)
    meta = model.metadata_props.add()
    meta.key = "tags"
    meta.value = " ".join(tags)
    onnx.save(model, args.output_path)


if __name__ == "__main__":
    main()
