#!/usr/bin/env python3
import argparse
import contextlib
import logging
import datasets
import enum
import sqlite3
from identifur import models
from identifur.data import E621DataModule
import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO)


class Category(enum.Enum):
    GENERAL = 0
    ARTIST = 1
    COPYRIGHT = 3
    CHARACTER = 4
    SPECIES = 5
    INVALID = 6
    META = 7
    LORE = 8


def load_tags(db, min_post_count):
    with contextlib.closing(db.cursor()) as cur:
        cur.execute(
            "SELECT id, name FROM tags WHERE post_count >= ? AND category IN (?, ?, ?, ?)",
            [
                min_post_count,
                Category.GENERAL.value,
                Category.CHARACTER.value,
                Category.COPYRIGHT.value,
                Category.SPECIES.value,
            ],
        )
        return list(cur)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--base-model", default="vit_l_16")
    argparser.add_argument("--tags-path", default="tags")
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

    db = sqlite3.connect(f"file:{args.data_db}?mode=ro", uri=True)
    tags = load_tags(db, args.tag_min_post_count)
    with open(args.tags_path, "wt", encoding="utf-8") as f:
        for _, name in tags:
            f.write(name)
            f.write("\n")

    ds = datasets.load_dataset(
        "hf/e621_samples.py",
        data_db_path=args.data_db,
        split="train",
    )

    dm = E621DataModule(
        dataset=ds,
        tags=tags,
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
        model=model,
        weights="DEFAULT",
        num_labels=len(tags) + 3,
        requires_grad=False,
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
