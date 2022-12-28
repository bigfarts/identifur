#!/usr/bin/env python3
import huggingface_hub
import argparse
import datasets
import re

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--dls-db", default="dls.db")
    argparser.add_argument("--images-path", default="images")
    argparser.add_argument("--hub-repo-id", default=None)
    argparser.add_argument("--num-processes", default=1, type=int)
    args = argparser.parse_args()

    date = re.match(r"^data-(\d{4}-\d{2}-\d{2})\.db$", args.data_db).group(1)

    repo_id = args.hub_repo_id
    if repo_id is None:
        api = huggingface_hub.HfApi()
        repo_id = f"{api.whoami()['name']}/e621_{date}"

    ds = datasets.load_dataset(
        "./hf/e621.py",
        name=date,
        num_proc=args.num_processes,
        db_path=args.data_db,
        dls_db_path=args.dls_db,
        images_path=args.images_path,
    )

    ds.push_to_hub(repo_id, private=True)
