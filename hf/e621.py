import contextlib
import struct
import logging
import os
import datasets
import sqlite3
import mmh3
from PIL import Image


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


_DESCRIPTION = "TODO"

_DB_PATH = "data-2022-12-26.db"
_DLS_DB_PATH = "dls.db"
_IMAGES_PATH = "images"
_NUM_SHARDS = 1024


class E621Dataset(datasets.GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "image": datasets.Image(),
                    "tags": datasets.features.Sequence(datasets.Value("string")),
                    "rating": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "db_path": _DB_PATH,
                    "images_path": _IMAGES_PATH,
                    "shard_ids": list(range(_NUM_SHARDS)),
                },
            )
        ]

    def _generate_examples(self, db_path, images_path, shard_ids):
        shard_ids = set(shard_ids)
        db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        db.execute("ATTACH DATABASE ? AS dls", [f"file:{_DLS_DB_PATH}?mode=ro"])

        with contextlib.closing(db.cursor()) as cur:
            cur.execute(
                "SELECT posts.id, posts.tag_string, posts.rating FROM dls.downloaded INNER JOIN posts ON dls.downloaded.post_id = posts.id"
            )
            for id, tag_string, rating in cur:
                if mmh3.hash(struct.pack(">Q", id)) % _NUM_SHARDS not in shard_ids:
                    continue

                fsid = format_split_id(split_id(id))
                try:
                    img = Image.open(os.path.join(images_path, *fsid))
                except Exception:
                    logging.exception("failed to load image %s", id)
                    continue

                tags = tag_string.split(" ")

                yield id, {
                    "image": img,
                    "tags": tags,
                    "rating": rating,
                }
