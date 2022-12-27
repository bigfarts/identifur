import enum
import contextlib
import torch
import os
from torch.utils.data import Dataset
from PIL import Image


class Category(enum.Enum):
    GENERAL = 0
    ARTIST = 1
    COPYRIGHT = 3
    CHARACTER = 4
    SPECIES = 5
    INVALID = 6
    META = 7
    LORE = 8


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


def load_tags(db, min_post_count):
    with contextlib.closing(db.cursor()) as cur:
        cur.execute(
            "SELECT name FROM tags WHERE post_count >= ? AND category IN (?, ?, ?, ?)",
            [
                min_post_count,
                Category.GENERAL.value,
                Category.CHARACTER.value,
                Category.COPYRIGHT.value,
                Category.SPECIES.value,
            ],
        )
        return [name for name, in cur] + [f"rating: {r}" for r in "sqe"]


class E621Dataset(Dataset):
    def __init__(self, post_ids, db, tags, dataset_path="dataset"):
        self.post_ids = post_ids
        self.db = db
        self.tags = tags
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.post_ids)

    def __getitem__(self, index):
        id = self.post_ids[index]

        fsid = format_split_id(split_id(id))
        img = Image.open(
            os.path.join(self.dataset_path, *fsid),
            formats=["JPEG", "PNG"],
        )

        with contextlib.closing(self.db.cursor()) as cur:
            cur.execute("SELECT tag_string, rating FROM posts WHERE id = ?", [id])
            (tag_string, rating) = cur.fetchone()

        tags = set(tag_string.split(" "))
        tags.add(f"rating: {rating}")

        img = img.convert("RGB")
        return (
            img,
            torch.tensor(
                [1 if tag in tags else 0 for tag in self.tags],
                dtype=torch.float32,
            ),
        )
