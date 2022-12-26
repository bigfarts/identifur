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


async def fetch(output_path, db, data_db, id, fetch_full_image):
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
                file_ext = "jpg"

        if resp is None:
            resp = await session.get(
                f"https://static1.e621.net/data/{split_path}.{file_ext}"
            )

        if resp.status == 404:
            logging.warn("%d not found, skipping", id)
            db.execute("UPDATE downloads SET downloaded = true WHERE id = ?", [id])
            db.commit()
            return

        resp.raise_for_status()

        with open(
            os.path.join(output_path, *fsid[:-1], f"{fsid[-1]}.{file_ext}"), "wb"
        ) as f:
            async for data in resp.content.iter_any():
                f.write(data)

        db.execute("UPDATE downloads SET downloaded = true WHERE id = ?", [id])
        db.commit()


async def worker(output_path, db, data_db, queue, fetch_full_image):
    try:
        while True:
            id = await queue.get()
            await guarded_fetch(output_path, db, data_db, id, fetch_full_image)
            queue.task_done()
    except asyncio.CancelledError:
        return


async def guarded_fetch(output_path, db, data_db, id, fetch_full_image):
    try:
        await fetch(output_path, db, data_db, id, fetch_full_image)
    except Exception:
        logging.exception("failed to download %d", id)


async def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--dls-db", default="dls.db")
    argparser.add_argument("--num-workers", default=16, type=int)
    argparser.add_argument("--output-path", default="dataset")
    argparser.add_argument("--fetch-full-image", default=False, action="store_true")
    args = argparser.parse_args()

    queue = asyncio.Queue(args.num_workers)

    db = sqlite3.connect(args.dls_db)
    data_db = sqlite3.connect(args.data_db)

    workers = [
        asyncio.create_task(
            worker(args.output_path, db, data_db, queue, args.fetch_full_image)
        )
        for _ in range(args.num_workers)
    ]

    cur = db.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM downloads WHERE NOT downloaded")
        (n,) = cur.fetchone()
    finally:
        cur.close()

    async def leader():
        cur = db.cursor()
        try:
            cur.execute("SELECT id FROM downloads WHERE NOT downloaded")

            pbar = tqdm(cur, total=n)
            for (id,) in pbar:
                pbar.set_description(f"{id}")
                await queue.put(id)
        finally:
            cur.close()
        await queue.join()

        for worker in workers:
            worker.cancel()

    await asyncio.gather(*([leader()] + workers))


if __name__ == "__main__":
    asyncio.run(main())
