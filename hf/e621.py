import contextlib
import struct
import logging
import os
import datasets
import sqlite3
import mmh3


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


_NUM_SHARDS = 1024


class E621Config(datasets.BuilderConfig):
    def __init__(
        self, data_db_path, images_path="images", dls_db_path="dls.db", **kwargs
    ):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)
        self.data_db_path = data_db_path
        self.images_path = images_path
        self.dls_db_path = dls_db_path


class E621Dataset(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = E621Config

    def _info(self):
        return datasets.DatasetInfo(
            description="""
All images of all ratings from e621.net from the date it was generated, at sample resolution where possible.

Note that this dataset excludes images that are, at the time of scraping:
- pending
- tagged with tags indicating that it is illegal to possess in most jurisdictions
""",
            features=datasets.Features(
                {
                    "id": datasets.Value("uint64"),
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
                    "shard_ids": list(range(_NUM_SHARDS)),
                },
            )
        ]

    def _generate_examples(self, shard_ids):
        shard_ids = set(shard_ids)
        db = sqlite3.connect(f"file:{self.config.data_db_path}?mode=ro", uri=True)
        db.execute(
            "ATTACH DATABASE ? AS dls", [f"file:{self.config.dls_db_path}?mode=ro"]
        )

        with contextlib.closing(db.cursor()) as cur:
            cur.execute(
                "SELECT posts.id, posts.tag_string, posts.rating FROM dls.downloaded INNER JOIN posts ON dls.downloaded.post_id = posts.id"
            )
            for id, tag_string, rating in cur:
                if (
                    len(shard_ids) != _NUM_SHARDS
                    and mmh3.hash(struct.pack(">Q", id)) % _NUM_SHARDS not in shard_ids
                ):
                    continue

                fsid = format_split_id(split_id(id))
                try:
                    with open(os.path.join(self.config.images_path, *fsid), "rb") as f:
                        buf = f.read()
                except Exception:
                    logging.exception("failed to load image %s", id)
                    continue

                yield id, {
                    "id": id,
                    "image": {"bytes": buf},
                    "tags": tag_string.split(" "),
                    "rating": rating,
                }
