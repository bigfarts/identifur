#!/usr/bin/env python3
import argparse
import enum
import struct
from tqdm import tqdm
import os
import logging
import contextlib
import sqlite3

logging.basicConfig(level=logging.INFO)


class Category(enum.Enum):
    GENERAL = 0
    ARTIST = 1
    COPYRIGHT = 3
    CHARACTER = 4
    SPECIES = 5
    INVALID = 6
    META = 7
    LORE = 8


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


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--dls-db", default="dls.db")
    argparser.add_argument("--dataset-path", default="dataset")
    argparser.add_argument("--tag-min-post-count", default=2500, type=int)
    args = argparser.parse_args()

    db = sqlite3.connect(f"file:{args.data_db}?mode=ro", uri=True)
    tags = list(load_tags(db, args.tag_min_post_count))
    logging.info("loaded %d tags", len(tags))

    meta_path = os.path.join(args.dataset_path, "_meta")
    try:
        os.makedirs(meta_path)
    except FileExistsError:
        pass

    with open(os.path.join(meta_path, "tags"), "wt", encoding="utf-8") as f:
        for tag in tags:
            f.write(f"{tag}\n")

    with open(os.path.join(meta_path, "index"), "wb") as f:
        db.execute("ATTACH DATABASE ? AS dls", [f"file:{args.dls_db}?mode=ro"])

        with contextlib.closing(db.cursor()) as cur:
            cur.execute("SELECT COUNT(*) FROM dls.downloaded")
            (n,) = cur.fetchone()

        with contextlib.closing(db.cursor()) as cur:
            cur.execute(
                "SELECT posts.id, posts.tag_string, posts.rating FROM dls.downloaded INNER JOIN posts ON dls.downloaded.post_id = posts.id"
            )
            for id, tag_string, rating in tqdm(cur, total=n):
                post_tags = set(tag_string.split(" "))
                post_tags.add(f"rating: {rating}")

                f.write(
                    struct.pack("Q", id)
                    + bytes(True if tag in post_tags else False for tag in tags)
                )


if __name__ == "__main__":
    main()
