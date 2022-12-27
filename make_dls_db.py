#!/usr/bin/env python3
import argparse
import contextlib
import logging
import csv
import sqlite3
from tqdm import tqdm

csv.field_size_limit(2147483647)

logging.basicConfig(level=logging.INFO)

RATINGS = "sqe"


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--blacklisted-tags", default="loli,shota")
    argparser.add_argument("--minimum-score", default=0, type=int)
    argparser.add_argument("--minimum-tag-count", default=0, type=int)
    argparser.add_argument("--maximum-rating", default="e")
    argparser.add_argument("--output-db-path", default="dls.db")
    args = argparser.parse_args()

    blacklisted_tags = set(
        args.blacklisted_tags.split(",") if args.blacklisted_tags else []
    )

    with (
        sqlite3.connect(args.output_db_path) as db,
        sqlite3.connect(f"file:{args.data_db}?mode=ro", uri=True) as data_db,
    ):
        db.executescript(
            """
        CREATE TABLE IF NOT EXISTS pending
            ( post_id INTEGER NOT NULL
            )
        STRICT;

        CREATE TABLE IF NOT EXISTS downloaded
            ( post_id INTEGER NOT NULL
            )
        STRICT;
        """
        )

        allowed_ratings = RATINGS[: RATINGS.index(args.maximum_rating) + 1]

        logging.info(
            "making dls db\nblacklisted tags: %s\nminimum score: %d\nminimum tag count: %d\nallowed ratings: %s",
            blacklisted_tags,
            args.minimum_score,
            args.minimum_tag_count,
            allowed_ratings,
        )

        with contextlib.closing(data_db.cursor()) as cur:
            cur.execute("SELECT COUNT(*) FROM posts")
            (n,) = cur.fetchone()

        with contextlib.closing(data_db.cursor()) as cur:
            cur.execute(
                "SELECT id, tag_string FROM posts WHERE NOT is_deleted AND NOT is_pending AND score >= ? AND INSTR(?, rating)",
                [args.minimum_score, allowed_ratings],
            )
            for id, tag_string in tqdm(cur, total=n):
                tags = tag_string.split(" ")
                if len(tags) < args.minimum_tag_count or any(
                    tag in blacklisted_tags for tag in tags
                ):
                    continue

                db.execute("""INSERT OR IGNORE INTO pending(post_id) VALUES(?)""", [id])


if __name__ == "__main__":
    main()
