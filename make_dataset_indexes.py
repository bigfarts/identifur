import argparse
import enum
from tqdm import tqdm
import os
import logging
import contextlib
import pyarrow as pa
from pyarrow import dataset
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

    with sqlite3.connect(f"file:{args.data_db}?mode=ro", uri=True) as db:
        tags = load_tags(db, args.tag_min_post_count)
        logging.info("loaded %d tags", len(tags))

    with open(os.path.join(args.dataset_path, "_tags"), "wt", encoding="utf-8") as f:
        for tag in tags:
            f.write(f"{tag}\n")

    schema = pa.schema(
        [
            ("post_id", pa.uint64()),
            ("labels", pa.list_(pa.bool_(), len(tags))),
        ]
    )

    def data_iter():
        with sqlite3.connect(f"file:{args.data_db}?mode=ro", uri=True) as db:
            db.execute("ATTACH DATABASE ? AS dls", [f"file:{args.dls_db}?mode=ro"])

            with contextlib.closing(db.cursor()) as cur:
                cur.execute(
                    "SELECT id, tag_string, rating FROM posts INNER JOIN dls.downloaded ON dls.downloaded.post_id = posts.id"
                )
                for id, tag_string, rating in tqdm(cur):
                    post_tags = set(tag_string.split(" "))
                    post_tags.add(f"rating: {rating}")

                    yield pa.RecordBatch.from_pylist(
                        [
                            {
                                "post_id": id,
                                "labels": [
                                    True if tag in post_tags else False for tag in tags
                                ],
                            }
                        ],
                        schema=schema,
                    )

    dataset.write_dataset(
        data_iter(),
        os.path.join(args.dataset_path, "_posts"),
        format="arrow",
        schema=schema,
    )


if __name__ == "__main__":
    main()
