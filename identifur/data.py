import enum
import torch
import logging
import os
from torch.utils.data import Dataset
import sqlite3
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


def format_id(sid):
    return tuple(f"{p:03}" for p in sid)


def load_tags(db, min_post_count):
    cur = db.cursor()
    try:
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
        return [name for name, in cur]
    finally:
        cur.close()


class E621Dataset(Dataset):
    def __init__(self, db, tags, dataset_path="dataset", max_id=None):
        self.db = db
        self.tags = tags
        self.dataset_path = dataset_path
        self.max_id = max_id

        if self.max_id is None:
            cur = self.db.cursor()
            try:
                cur.execute("SELECT MAX(id) FROM posts")
                (self.max_id,) = cur.fetchone()
            finally:
                cur.close()

    def __len__(self):
        return self.max_id + 1

    def __getitem__(self, index):
        if index > self.max_id:
            raise IndexError

        try:
            img = None
            fsid = format_id(split_id(index))
            for ext in (".jpg", ".png"):
                path = os.path.join(self.dataset_path, *fsid[:-1], fsid[-1] + ext)
                try:
                    f = open(path, "rb")
                except FileNotFoundError:
                    continue

                img = Image.open(f)
                img.load()

            if img is None:
                return None

            cur = self.db.cursor()
            try:
                cur.execute("SELECT tag_string FROM posts WHERE id = ?", [index])
                row = cur.fetchone()
            finally:
                cur.close()

            if row is None:
                return None

            (tag_string,) = row

            tags = set(tag_string.split(" "))

            with img:
                return (
                    img.convert("RGB"),
                    torch.tensor(
                        [1 if tag in tags else 0 for tag in self.tags],
                        dtype=torch.float32,
                    ),
                )
        except Exception:
            logging.exception("failed to open index %d", index)
            return None
