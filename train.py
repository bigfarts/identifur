#!/usr/bin/env python3
import argparse
import contextlib
import sqlite3
import logging
from identifur import models
from identifur.data import load_tags, E621DataModule
import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO)


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--dls-db", default="dls.db")
    argparser.add_argument("--base-model", default="vit_l_16")
    argparser.add_argument("--dataset-path", default="dataset")
    argparser.add_argument("--random-split-seed", default=42, type=int)
    argparser.add_argument("--batch-size", default=64, type=int)
    argparser.add_argument("--tag-min-post-count", default=2500, type=int)
    argparser.add_argument("--train-data-split", default=0.6, type=float)
    argparser.add_argument("--validation-data-split", default=0.2, type=float)
    argparser.add_argument("--test-data-split", default=0.2, type=float)
    argparser.add_argument("--max-epochs", default=10, type=int)
    argparser.add_argument("--num-workers", default=0, type=int)
    argparser.add_argument("--auto-lr-find", default=False, action="store_true")
    args = argparser.parse_args()

    with sqlite3.connect(f"file:{args.dls_db}?mode=ro", uri=True) as dls_db:
        with contextlib.closing(dls_db.cursor()) as cur:
            cur.execute("SELECT post_id FROM downloaded")
            post_ids = [post_id for post_id, in cur]
    logging.info("loaded %d post IDs", len(post_ids))

    with sqlite3.connect(f"file:{args.data_db}?mode=ro", uri=True) as db:
        tags = load_tags(db, args.tag_min_post_count)
        logging.info("loaded %d tags", len(tags))

        model, input_size = models.MODELS[args.base_model]

        dm = E621DataModule(
            post_ids=post_ids,
            db=db,
            tags=tags,
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
            model=model(pretrained=True, requires_grad=False, num_classes=len(tags)),
            num_labels=len(tags),
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
