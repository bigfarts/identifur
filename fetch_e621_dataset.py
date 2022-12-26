#!/usr/bin/env python3
import argparse
import asyncio
from tqdm import tqdm
import logging
import aiohttp
import sqlite3
import os
from identifur.data import split_id, format_id

logging.basicConfig(level=logging.INFO)


async def fetch(output_path, db, data_db, id):
    fsid = format_id(split_id(id))
    try:
        os.makedirs(os.path.join(output_path, *fsid[:-1]))
    except FileExistsError:
        pass

    cur = data_db.cursor()
    try:
        cur.execute("SELECT md5, file_ext FROM posts WHERE id = ?", [id])
        row = cur.fetchone()
    finally:
        cur.close()

    if row is None:
        return

    md5, file_ext = row

    if file_ext not in ("jpg", "png"):
        return

    split_path = f"{md5[0:2]}/{md5[2:4]}/{md5}"

    async with aiohttp.ClientSession() as session:
        # Try get the sample first, because it's cheaper.
        resp = await session.get(
            f"https://static1.e621.net/data/sample/{split_path}.jpg"
        )
        if resp.status == 404:
            resp = await session.get(
                f"https://static1.e621.net/data/{split_path}.{file_ext}"
            )
        else:
            file_ext = "jpg"

        if resp.status == 404:
            logging.warn("%d not found, skipping", id)
            db.execute("UPDATE downloads SET downloaded = true WHERE id = ?", [id])
            db.commit()
            return

        resp.raise_for_status()

        with open(
            os.path.join(output_path, *fsid[:-1], f"{fsid[-1]}.{file_ext}"), "wb"
        ) as f:
            f.write(await resp.read())

        db.execute("UPDATE downloads SET downloaded = true WHERE id = ?", [id])
        db.commit()


async def worker(output_path, db, data_db, queue):
    while True:
        id = await queue.get()
        await guarded_fetch(output_path, db, data_db, id)


async def guarded_fetch(output_path, db, data_db, id):
    try:
        await fetch(output_path, db, data_db, id)
    except Exception:
        logging.exception("failed to download %d", id)


async def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--dls-db", default="dls.db")
    argparser.add_argument("--num-workers", default=16, type=int)
    argparser.add_argument("--output-path", default="dataset")
    args = argparser.parse_args()

    queue = asyncio.Queue(args.num_workers)

    db = sqlite3.connect(args.dls_db)
    data_db = sqlite3.connect(args.data_db)

    workers = [
        asyncio.create_task(worker(args.output_path, db, data_db, queue))
        for _ in range(args.num_workers)
    ]

    cur = db.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM downloads WHERE NOT downloaded")
        (n,) = cur.fetchone()
    finally:
        cur.close()

    try:
        cur = db.cursor()
        cur.execute("SELECT id FROM downloads WHERE NOT downloaded")

        pbar = tqdm(cur, total=n)
        for (id,) in pbar:
            pbar.set_description(f"{id}")
            await queue.put(id)
    finally:
        cur.close()


if __name__ == "__main__":
    asyncio.run(main())
