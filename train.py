#!/usr/bin/env python3
import argparse
import logging
from identifur import models
from identifur.data import E621DataModule, load_tags
import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--base-model", default="vit_l_16")
    argparser.add_argument("--dataset-path", default="./hf/e621.py")
    argparser.add_argument("--random-split-seed", default=42, type=int)
    argparser.add_argument("--batch-size", default=64, type=int)
    argparser.add_argument("--train-data-split", default=0.6, type=float)
    argparser.add_argument("--validation-data-split", default=0.2, type=float)
    argparser.add_argument("--test-data-split", default=0.2, type=float)
    argparser.add_argument("--max-epochs", default=10, type=int)
    argparser.add_argument("--num-workers", default=0, type=int)
    argparser.add_argument("--auto-lr-find", default=False, action="store_true")
    argparser.add_argument("--tag-min-post-count", default=2500, type=int)
    args = argparser.parse_args()

    model, input_size = models.MODELS[args.base_model]

    tags = list(load_tags(args.dataset_path))
    dm = E621DataModule(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        splits=[
            args.train_data_split,
            args.validation_data_split,
            args.test_data_split,
        ],
        split_seed=args.random_split_seed,
        input_size=input_size,
        num_workers=args.num_workers,
    )

    model = models.LitModel(
        model=model, weights="DEFAULT", num_labels=len(tags), requires_grad=False
    )

    trainer = pl.Trainer(
        auto_lr_find=args.auto_lr_find,
        max_epochs=args.max_epochs,
        accelerator="gpu",
    )
    trainer.fit(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
