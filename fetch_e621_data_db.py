#!/usr/bin/env python3
import argparse
import logging
import sqlite3
from tqdm import tqdm
import gzip
import csv
import requests

logging.basicConfig(level=logging.INFO)

csv.field_size_limit(2147483647)


def init_db(db):
    db.executescript(
        """
CREATE TABLE IF NOT EXISTS posts
    ( id INTEGER PRIMARY KEY
    , uploader_id INTEGER NOT NULL
    , created_at TEXT NOT NULL
    , md5 TEXT NOT NULL
    , source TEXT
    , rating TEXT NOT NULL
    , image_width INTEGER NOT NULL
    , image_height INTEGER NOT NULL
    , tag_string TEXT NOT NULL
    , locked_tags TEXT NOT NULL
    , fav_count INTEGER NOT NULL
    , file_ext TEXT NOT NULL
    , parent_id INTEGER
    , change_seq INTEGER NOT NULL
    , approver_id INTEGER
    , file_size INTEGER NOT NULL
    , comment_count INTEGER NOT NULL
    , description TEXT NOT NULL
    , duration REAL
    , updated_at TEXT NOT NULL
    , is_deleted INTEGER NOT NULL  -- actually bool
    , is_pending INTEGER NOT NULL  -- actually bool
    , is_flagged INTEGER NOT NULL  -- actually bool
    , score INTEGER NOT NULL
    , up_score INTEGER NOT NULL
    , down_score INTEGER NOT NULL
    , is_rating_locked INTEGER NOT NULL  -- actually bool
    , is_status_locked INTEGER NOT NULL  -- actually bool
    , is_note_locked INTEGER NOT NULL  -- actually bool
    )
STRICT;

CREATE TABLE IF NOT EXISTS tags
    ( id INTEGER PRIMARY KEY
    , name TEXT NOT NULL
    , category INTEGER NOT NULL
    , post_count INTEGER NOT NULL
    )
STRICT;
"""
    )


def fetch_posts(db, date):
    req = requests.get(f"https://e621.net/db_export/posts-{date}.csv.gz", stream=True)
    req.raise_for_status()

    with gzip.open(req.raw, "rt", encoding="utf-8") as f:
        csvr = csv.DictReader(f)
        for row in tqdm(csvr, desc="posts"):
            id = int(row["id"])
            uploader_id = int(row["uploader_id"])
            created_at = row["created_at"]
            md5 = row["md5"]
            source = row["source"] if row["source"] else None
            rating = row["rating"]
            tag_string = row["tag_string"]
            image_width = int(row["image_width"])
            image_height = int(row["image_height"])
            locked_tags = row["locked_tags"]
            fav_count = int(row["fav_count"])
            file_ext = row["file_ext"]
            parent_id = int(row["parent_id"]) if row["parent_id"] else None
            change_seq = int(row["change_seq"])
            approver_id = int(row["approver_id"]) if row["approver_id"] else None
            file_size = int(row["file_size"])
            comment_count = int(row["comment_count"])
            description = row["description"]
            duration = float(row["duration"]) if row["duration"] else None
            updated_at = row["updated_at"]
            is_deleted = row["is_deleted"] == "t"
            is_pending = row["is_pending"] == "t"
            is_flagged = row["is_flagged"] == "t"
            score = int(row["score"])
            up_score = int(row["up_score"])
            down_score = int(row["down_score"])
            is_rating_locked = row["is_rating_locked"] == "t"
            is_status_locked = row["is_status_locked"] == "t"
            is_note_locked = row["is_note_locked"] == "t"

            db.execute(
                """
                INSERT OR IGNORE INTO posts
                    ( id
                    , uploader_id
                    , created_at
                    , md5
                    , source
                    , rating
                    , image_width
                    , image_height
                    , tag_string
                    , locked_tags
                    , fav_count
                    , file_ext
                    , parent_id
                    , change_seq
                    , approver_id
                    , file_size
                    , comment_count
                    , description
                    , duration
                    , updated_at
                    , is_deleted
                    , is_pending
                    , is_flagged
                    , score
                    , up_score
                    , down_score
                    , is_rating_locked
                    , is_status_locked
                    , is_note_locked
                    )
                VALUES
                    ( ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    , ?
                    )
                """,
                [
                    id,
                    uploader_id,
                    created_at,
                    md5,
                    source,
                    rating,
                    image_width,
                    image_height,
                    tag_string,
                    locked_tags,
                    fav_count,
                    file_ext,
                    parent_id,
                    change_seq,
                    approver_id,
                    file_size,
                    comment_count,
                    description,
                    duration,
                    updated_at,
                    is_deleted,
                    is_pending,
                    is_flagged,
                    score,
                    up_score,
                    down_score,
                    is_rating_locked,
                    is_status_locked,
                    is_note_locked,
                ],
            )


def fetch_tags(db, date):
    req = requests.get(f"https://e621.net/db_export/tags-{date}.csv.gz", stream=True)
    req.raise_for_status()

    with gzip.open(req.raw, "rt", encoding="utf-8") as f:
        csvr = csv.DictReader(f)
        for row in tqdm(csvr, desc="tags"):
            id = int(row["id"])
            name = row["name"]
            post_count = int(row["post_count"])
            category = int(row["category"])

            db.execute(
                "INSERT OR IGNORE INTO tags (id, name, category, post_count) VALUES (?, ?, ?, ?)",
                [id, name, category, post_count],
            )


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("date")
    argparser.add_argument("--output-path", default=None)
    args = argparser.parse_args()

    output_path = args.output_path
    if output_path is None:
        output_path = f"data-{args.date}.db"

    db = sqlite3.connect(output_path)
    init_db(db)
    fetch_posts(db, args.date)
    fetch_tags(db, args.date)
    db.commit()


if __name__ == "__main__":
    main()
