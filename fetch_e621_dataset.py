#!/usr/bin/env python3
import argparse
import humanfriendly
import asyncio
import contextlib
from tqdm import tqdm
import logging
import aiohttp
import sqlite3
import os
from identifur.id import split_id, format_split_id

logging.basicConfig(level=logging.INFO)


class Stats:
    def __init__(self):
        self.bytes_received = 0


async def fetch(session, stats, output_path, db, data_db, id, fetch_full_image):
    fsid = format_split_id(split_id(id))
    try:
        os.makedirs(os.path.join(output_path, *fsid[:-1]))
    except FileExistsError:
        pass

    with contextlib.closing(data_db.cursor()) as cur:
        cur.execute("SELECT md5, file_ext FROM posts WHERE id = ?", [id])
        row = cur.fetchone()

    if row is None:
        return

    md5, file_ext = row

    if file_ext not in ("jpg", "png"):
        return

    split_path = f"{md5[0:2]}/{md5[2:4]}/{md5}"

    resp = None
    if not fetch_full_image:
        # Prefer samples, since we don't actually process the whole thing anyway.
        resp = await session.get(
            f"https://static1.e621.net/data/sample/{split_path}.jpg"
        )
        if resp.status == 404:
            resp = None
        else:
            resp.raise_for_status()

    if resp is None:
        resp = await session.get(
            f"https://static1.e621.net/data/{split_path}.{file_ext}"
        )

    if resp.status == 404:
        logging.warn("%d not found, skipping", id)
        db.execute("INSERT INTO visited(post_id) VALUES(?)", [id])
        db.commit()
        return

    resp.raise_for_status()

    path = os.path.join(output_path, *fsid)
    incomplete_path = path + ".incomplete"

    with open(incomplete_path, "wb") as f:
        async for data in resp.content.iter_any():
            stats.bytes_received += len(data)
            f.write(data)
    os.rename(incomplete_path, path)

    db.execute("INSERT INTO downloaded(post_id) VALUES(?)", [id])
    db.execute("INSERT INTO visited(post_id) VALUES(?)", [id])
    db.commit()


async def worker(session, stats, output_path, db, data_db, queue, fetch_full_image):
    try:
        while True:
            id = await queue.get()
            await guarded_fetch(
                session, stats, output_path, db, data_db, id, fetch_full_image
            )
            queue.task_done()
    except asyncio.CancelledError:
        return


async def guarded_fetch(session, stats, output_path, db, data_db, id, fetch_full_image):
    try:
        await fetch(session, stats, output_path, db, data_db, id, fetch_full_image)
    except Exception:
        logging.exception("failed to download %d", id)


def init_db(db):
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS selected
            ( post_id INTEGER PRIMARY KEY
            )
        STRICT;

        CREATE TABLE IF NOT EXISTS visited
            ( post_id INTEGER PRIMARY KEY
            )
        STRICT;

        CREATE TABLE IF NOT EXISTS downloaded
            ( post_id INTEGER PRIMARY KEY
            )
        STRICT;
        """
    )


def rebuild_selected_table(
    db,
    data_db,
    minimum_score=0,
    allowed_ratings="sqe",
    blacklisted_tags=frozenset(),
    minimum_tag_count=0,
):
    with contextlib.closing(data_db.cursor()) as cur:
        cur.execute("SELECT COUNT(*) FROM posts")
        (n,) = cur.fetchone()

    with contextlib.closing(data_db.cursor()) as cur:
        cur.execute(
            "SELECT id, tag_string FROM posts WHERE NOT is_deleted AND NOT is_pending AND score >= ? AND INSTR(?, rating)",
            [minimum_score, allowed_ratings],
        )
        for id, tag_string in tqdm(cur, total=n):
            tags = tag_string.split(" ")
            if len(tags) < minimum_tag_count or any(
                tag in blacklisted_tags for tag in tags
            ):
                continue

            db.execute("""INSERT OR IGNORE INTO selected(post_id) VALUES(?)""", [id])


RATINGS = "sqe"


async def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--dls-db", default="dls.db")
    argparser.add_argument(
        "--rebuild-selected-table", default=False, action="store_true"
    )
    argparser.add_argument("--num-workers", default=16, type=int)
    argparser.add_argument("--output-path", default="dataset")
    argparser.add_argument("--fetch-full-image", default=False, action="store_true")
    argparser.add_argument("--blacklisted-tags", default="loli,shota")
    argparser.add_argument("--minimum-score", default=0, type=int)
    argparser.add_argument("--minimum-tag-count", default=0, type=int)
    argparser.add_argument("--maximum-rating", default="e")
    args = argparser.parse_args()

    queue = asyncio.Queue(args.num_workers)

    with (
        sqlite3.connect(args.dls_db) as db,
        sqlite3.connect(f"file:{args.data_db}?mode=ro", uri=True) as data_db,
    ):
        init_db(db)

        should_rebuild_selected_table = args.rebuild_selected_table
        if not should_rebuild_selected_table:
            with contextlib.closing(db.cursor()) as cur:
                cur.execute("SELECT NOT EXISTS(SELECT 1 FROM selected)")
                (should_rebuild_selected_table,) = cur.fetchone()

        if should_rebuild_selected_table:
            blacklisted_tags = frozenset(
                args.blacklisted_tags.split(",") if args.blacklisted_tags else []
            )
            allowed_ratings = RATINGS[: RATINGS.index(args.maximum_rating) + 1]
            logging.info(
                "selected table was either empty or rebuild was requested\nblacklisted tags: %s\nminimum score: %d\nminimum tag count: %d\nallowed ratings: %s",
                blacklisted_tags,
                args.minimum_score,
                args.minimum_tag_count,
                allowed_ratings,
            )
            rebuild_selected_table(
                db,
                data_db,
                args.minimum_score,
                allowed_ratings,
                blacklisted_tags,
                args.minimum_tag_count,
            )

        stats = Stats()
        async with aiohttp.ClientSession() as session:
            workers = [
                asyncio.create_task(
                    worker(
                        session,
                        stats,
                        args.output_path,
                        db,
                        data_db,
                        queue,
                        args.fetch_full_image,
                    )
                )
                for _ in range(args.num_workers)
            ]

            async def leader():
                with contextlib.closing(db.cursor()) as cur:
                    cur.execute(
                        "SELECT (SELECT COUNT(*) FROM selected) - (SELECT COUNT(*) FROM visited)"
                    )
                    (n,) = cur.fetchone()

                with contextlib.closing(db.cursor()) as cur:
                    cur.execute(
                        "SELECT selected.post_id FROM selected LEFT JOIN visited ON selected.post_id = visited.post_id WHERE visited.post_id IS NULL"
                    )

                    pbar = tqdm(cur, total=n)
                    for (id,) in pbar:
                        rate = stats.bytes_received / pbar.format_dict["elapsed"]
                        pbar.set_postfix(
                            {
                                "speed": f"{humanfriendly.format_size(rate, binary=True)}/s"
                            }
                        )
                        pbar.set_description(f"{id}")
                        await queue.put(id)

                await queue.join()

                for worker in workers:
                    worker.cancel()

            await asyncio.gather(*([leader()] + workers))


if __name__ == "__main__":
    asyncio.run(main())
