#!/usr/bin/env python3
import huggingface_hub
import argparse
import datasets
import re

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--date", default=None)
    argparser.add_argument("--hub-repo-id", default=None)
    argparser.add_argument("--num-processes", default=1, type=int)
    argparser.add_argument("--writer-batch-size", default=None, type=int)
    args = argparser.parse_args()

    date = args.date
    if date is None:
        date = re.match(r"^data-(\d{4}-\d{2}-\d{2})\.db$", args.data_db).group(1)

    repo_id = args.hub_repo_id
    if repo_id is None:
        api = huggingface_hub.HfApi()
        repo_id = f"{api.whoami()['name']}/e621_samples_{date}"

    ds = datasets.load_dataset(
        "./hf/e621_samples.py",
        name=date,
        num_proc=args.num_processes,
        data_db_path=args.data_db,
        writer_batch_size=args.writer_batch_size,
        split=datasets.Split.TRAIN,
    )

    # ds.save_to_disk("dataset")
    ds.push_to_hub(repo_id, private=True)
