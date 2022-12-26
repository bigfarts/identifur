import enum
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
    def __init__(self, dls_db, db, tags, dataset_path="dataset"):
        self.dls_db = dls_db
        self.db = db
        self.tags = tags
        self.dataset_path = dataset_path

        cur = self.dls_db.cursor()
        try:
            cur.execute("SELECT MAX(rowid) FROM downloaded")
            (self.n,) = cur.fetchone()
        finally:
            cur.close()

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        if index >= self.n:
            raise IndexError

        cur = self.dls_db.cursor()
        try:
            cur.execute("SELECT post_id FROM downloaded WHERE rowid = ?", [index + 1])
            (id,) = cur.fetchone()
        finally:
            cur.close()

        try:
            img = None
            fsid = format_id(split_id(id))
            for ext in (".jpg", ".png"):
                path = os.path.join(self.dataset_path, *fsid[:-1], fsid[-1] + ext)
                try:
                    f = open(path, "rb")
                except FileNotFoundError:
                    continue

                img = Image.open(f)

            if img is None:
                raise ValueError("image not found")

            cur = self.db.cursor()
            try:
                cur.execute("SELECT tag_string FROM posts WHERE id = ?", [id])
                (tag_string,) = cur.fetchone()
            finally:
                cur.close()

            tags = set(tag_string.split(" "))

            img = img.convert("RGB")
            # TODO: Handle errors here.

            return (
                img,
                torch.tensor(
                    [1 if tag in tags else 0 for tag in self.tags],
                    dtype=torch.float32,
                ),
            )
        except Exception as e:
            raise ValueError(f"failed to open {id}") from e
