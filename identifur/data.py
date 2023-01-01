import logging
import torchvision.transforms.functional as F
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms
from PIL import Image


def _image_without_transparency(img: Image.Image):
    if img.mode == "RGBA":
        img2 = Image.new("RGB", img.size, (0, 0, 0))
        img2.paste(img, mask=img.split()[3])
        return img2

    return img.convert("RGB")


class E621Dataset(Dataset):
    def __init__(self, tags, dataset):
        self.tags = tags
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        row = self.dataset[i]

        tags_set = set(row["tags"])

        return (
            _image_without_transparency(row["image"]),
            torch.tensor(
                [1.0 if id in tags_set else 0.0 for id, _ in self.tags]
                + [1.0 if row["rating"] == i else 0.0 for i in range(3)],
                dtype=torch.float32,
            ),
        )


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, "constant")


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
        tags,
        dataset,
        batch_size=32,
        splits=(0.6, 0.2, 0.2),
        split_seed=42,
        input_size=(224, 224),
        num_workers=0,
    ):
        super().__init__()
        self.tags = tags
        self.dataset = dataset
        self.batch_size = batch_size
        self.splits = splits
        self.split_seed = split_seed
        self.input_size = input_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        ds = SafeDataset(E621Dataset(self.tags, self.dataset))

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
                        SquarePad(),
                        transforms.Resize(self.input_size),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomRotation(degrees=180),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            TransformingDataset(
                self.val,
                transforms.Compose(
                    [
                        SquarePad(),
                        transforms.Resize(self.input_size),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            TransformingDataset(
                self.val,
                transforms.Compose(
                    [
                        SquarePad(),
                        transforms.Resize(self.input_size),
                        transforms.ToTensor(),
                    ]
                ),
            ),
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )
