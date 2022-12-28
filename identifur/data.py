import functools
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
        for line in f:
            yield line.rstrip("\n")


class Index:
    def __init__(self, dataset_path):
        self.file = open(os.path.join(dataset_path, "_meta", "index"), "rb")
        self.num_tags = sum(1 for _ in load_tags(dataset_path))
        self.mmap = mmap.mmap(self.file.fileno(), 0, mmap.MAP_PRIVATE, mmap.PROT_READ)

    @property
    def _record_size(self):
        return struct.calcsize("Q") + self.num_tags

    def __len__(self):
        return self.mmap.size() // self._record_size

    def __getitem__(self, i):
        record_size = struct.calcsize("Q") + self.num_tags
        offset = i * record_size
        (id,) = struct.unpack("Q", self.mmap[offset : offset + struct.calcsize("Q")])
        return id, torch.tensor(
            [
                x == True
                for x in self.mmap[
                    offset
                    + struct.calcsize("Q") : offset
                    + struct.calcsize("Q")
                    + self.num_tags
                ]
            ],
            dtype=torch.float32,
        )


class E621Dataset(Dataset):
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = dataset_path

    def __getstate__(self):
        return {
            k: v
            for k, v in self.__dict__.items()
            if not isinstance(
                getattr(self.__class__, k, None), functools.cached_property
            )
        }

    @functools.cached_property
    def _index(self):
        return Index(self.dataset_path)

    def image_for_id(self, id):
        fsid = format_split_id(split_id(id))
        return Image.open(os.path.join(self.dataset_path, *fsid)).convert("RGB")

    def __len__(self):
        return len(self._index)

    def __getitem__(self, i):
        id, labels = self._index[i]
        return (
            transforms.ToTensor()(self.image_for_id(id)),
            labels,
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
