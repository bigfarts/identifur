#!/usr/bin/env python3
import argparse
import contextlib
import torch
import sqlite3
import logging
from identifur import models
from identifur.data import E621Dataset, load_tags
from torch import optim, nn
from torchvision import transforms
from torchmetrics import Accuracy
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl

logging.basicConfig(level=logging.INFO)


class TransformingDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]
        if item is None:
            return None
        image, label = item
        return self.transform(image), label


class SafeDataset(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        while True:
            try:
                return self.ds[index]
            except Exception:
                logging.exception("failed to open %d, will pick next one", index)
            index = (index + 1) % len(self)


class E621DataModule(pl.LightningDataModule):
    def __init__(
        self,
        post_ids,
        db,
        tags,
        dataset_path="dataset",
        batch_size=32,
        splits=(0.6, 0.2, 0.2),
        split_seed=42,
        input_size=(224, 224),
        num_workers=0,
    ):
        super().__init__()
        self.post_ids = post_ids
        self.db = db
        self.tags = tags
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.splits = splits
        self.split_seed = split_seed
        self.input_size = input_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        ds = SafeDataset(
            E621Dataset(self.post_ids, self.db, self.tags, self.dataset_path)
        )

        self.train, self.val, self.test = random_split(
            ds,
            self.splits,
            generator=torch.Generator().manual_seed(self.split_seed),
        )

    def train_dataloader(self):
        return DataLoader(
            TransformingDataset(
                self.train,
                transforms.Compose(
                    [
                        transforms.Resize(self.input_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees=45),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            TransformingDataset(
                self.val,
                transforms.Compose(
                    [
                        transforms.Resize(self.input_size),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            TransformingDataset(
                self.val,
                transforms.Compose(
                    [
                        transforms.Resize(self.input_size),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class E621Module(pl.LightningModule):
    def __init__(self, model, num_labels, lr=2e-4):
        super().__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()
        self.accuracy = Accuracy(task="multilabel", num_labels=num_labels)

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

        return {"loss": loss, "outputs": y_pred, "y": y}

    def test_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        output = torch.cat([x["outputs"] for x in outputs], dim=0)

        ys = torch.cat([x["gt"] for x in outputs], dim=0)

        self.log("test/loss", loss)
        acc = self.accuracy(output, ys)
        self.log("test/acc", acc)

        self.test_ys = ys
        self.test_output = output


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

        model = E621Module(
            model=model(pretrained=True, requires_grad=False, num_classes=len(tags)),
            num_labels=len(tags),
        )

        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator="gpu",
        )
        trainer.fit(model, dm)
        trainer.test(model, dm)


if __name__ == "__main__":
    main()
