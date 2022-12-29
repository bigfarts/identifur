import contextlib
import datetime
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

_RATINGS = "sqe"


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

This includes the following additional metadata:
- post ID
- created at
- updated at
- tags (stored as IDs you can cross-reference from an e621 tags dump)
- rating (0 = safe, 1 = questionable, 2 = explicit)
- favorite count
- comment count
- up score
- down score

Note that this dataset excludes images that are, at the time of scraping:
- pending
- tagged with tags indicating that it is illegal to possess in most jurisdictions
""",
            features=datasets.Features(
                {
                    "id": datasets.Value("uint32"),
                    "created_at": datasets.Value("timestamp[us]"),
                    "updated_at": datasets.Value("timestamp[us]"),
                    "image": datasets.Image(),
                    "tags": datasets.features.Sequence(datasets.Value("uint32")),
                    "rating": datasets.Value("uint8"),
                    "fav_count": datasets.Value("uint32"),
                    "comment_count": datasets.Value("uint32"),
                    "up_score": datasets.Value("int32"),
                    "down_score": datasets.Value("int32"),
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

        with contextlib.closing(db.cursor()) as cur:
            cur.execute("SELECT id, name FROM tags")
            tag_ids = {name: id for id, name in cur}

        db.execute(
            "ATTACH DATABASE ? AS dls", [f"file:{self.config.dls_db_path}?mode=ro"]
        )

        with contextlib.closing(db.cursor()) as cur:
            cur.execute(
                "SELECT posts.id, posts.created_at, posts.updated_at, posts.tag_string, posts.rating, posts.fav_count, posts.comment_count, posts.up_score, posts.down_score FROM dls.downloaded INNER JOIN posts ON dls.downloaded.post_id = posts.id"
            )
            for (
                id,
                created_at,
                updated_at,
                tag_string,
                rating,
                fav_count,
                comment_count,
                up_score,
                down_score,
            ) in cur:
                if (
                    len(shard_ids) != _NUM_SHARDS
                    and mmh3.hash(struct.pack(">Q", id)) % _NUM_SHARDS not in shard_ids
                ):
                    continue

                created_at = (
                    datetime.datetime.strptime(created_at, "%Y-%m-%d %H:%M:%S.%f")
                    if created_at
                    else None
                )
                updated_at = (
                    datetime.datetime.strptime(updated_at, "%Y-%m-%d %H:%M:%S.%f")
                    if updated_at
                    else None
                )

                fsid = format_split_id(split_id(id))
                try:
                    with open(os.path.join(self.config.images_path, *fsid), "rb") as f:
                        buf = f.read()
                except Exception:
                    logging.exception("failed to load image %s", id)
                    continue

                yield id, {
                    "id": id,
                    "created_at": created_at,
                    "updated_at": updated_at,
                    "image": {"bytes": buf},
                    "tags": [tag_ids[tag] for tag in tag_string.split(" ")],
                    "rating": _RATINGS.index(rating),
                    "fav_count": fav_count,
                    "comment_count": comment_count,
                    "up_score": up_score,
                    "down_score": down_score,
                }
