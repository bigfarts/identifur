from pyarrow import dataset
import logging
import contextlib
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pytorch_lightning as pl
from torchvision import transforms


def split_id(id, depth=3, factor=1000):
    parts = []
    while depth > 0:
        parts.append(id % factor)
        id //= factor
        depth -= 1

    return tuple(reversed(parts))


def format_split_id(sid):
    parts = [f"{p:03}" for p in sid]
    parts[-1] = "".join(parts)
    return parts


def load_tags(dataset_path):
    with open(os.path.join(dataset_path, "_tags"), "rt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class E621Dataset(Dataset):
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.tags = load_tags(dataset_path)
        self.table = dataset.dataset(
            os.path.join(dataset_path, "_posts"), format="arrow"
        ).to_table()

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, index):
        fsid = format_split_id(split_id(self.table[0][index].as_py()))
        img = Image.open(
            os.path.join(self.dataset_path, *fsid),
            formats=["JPEG", "PNG"],
        )

        img = img.convert("RGB")
        return (
            img,
            torch.tensor(
                [v.as_py() for v in self.table[1][index]],
                dtype=torch.float32,
            ),
        )


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
        dataset_path="dataset",
        batch_size=32,
        splits=(0.6, 0.2, 0.2),
        split_seed=42,
        input_size=(224, 224),
        num_workers=0,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.splits = splits
        self.split_seed = split_seed
        self.input_size = input_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        ds = SafeDataset(E621Dataset(self.dataset_path))

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
