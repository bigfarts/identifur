import logging
import struct
import mmap
import torch
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import pytorch_lightning as pl
from torchvision import transforms
from .id import format_split_id, split_id


def load_tags(dataset_path):
    with open(os.path.join(dataset_path, "_meta", "tags"), "rt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


class E621Dataset(Dataset):
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path
        self.tags = load_tags(dataset_path)
        self.index_file = open(os.path.join(dataset_path, "_meta", "index"), "rb")
        self.index = mmap.mmap(
            self.index_file.fileno(), 0, mmap.MAP_PRIVATE, mmap.PROT_READ
        )
        self.record_size = struct.calcsize("Q") + len(self.tags)

    def id_for_index(self, index):
        offset = index * self.record_size
        (id,) = struct.unpack("Q", self.index[offset : offset + struct.calcsize("Q")])
        return id

    def labels_for_index(self, index):
        offset = index * self.record_size
        return torch.tensor(
            [
                x == True
                for x in self.index[
                    offset
                    + struct.calcsize("Q") : offset
                    + struct.calcsize("Q")
                    + len(self.tags)
                ]
            ],
            dtype=torch.float32,
        )

    def image_for_id(self, id):
        fsid = format_split_id(split_id(id))
        return Image.open(
            os.path.join(self.dataset_path, *fsid),
            formats=["JPEG", "PNG"],
        ).convert("RGB")

    def __len__(self):
        return self.index.size() // self._get_record_size()

    def __getitem__(self, index):
        id = self.id_for_index(index)
        return (
            transforms.ToTensor()(self.image_for_id(id)),
            self.labels_for_index(index),
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
                nn.Sequential(
                    transforms.Resize(self.input_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=180),
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
                nn.Sequential(
                    transforms.Resize(self.input_size),
                ),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            TransformingDataset(
                self.val,
                nn.Sequential(
                    transforms.Resize(self.input_size),
                ),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
