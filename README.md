# identifur

identifur tries to automatically tag furry art.

## fetching the dataset

identifur has a bunch of scripts you should use to fetch the dataset. the scripts fetch data from e621, so you're gonna end up with a lot of furry porn.

### 1. fetch the data DB

```sh
python3 fetch_e621_data_db.py $DATE
```

this generates a `data-$DATE.db` file for use with everything else. if you're not sure what to use for the date, check the files in https://e621.net/db_export/.

### 2. fetch the dataset

```sh
python3 fetch_e621_dataset.py data-$DATE.db
```

this will start populating the `dataset` directory. you can interrupt and restart the script at any time: current status is saved to `dls.db`. if you need to update the selections (e.g. due to a different minimum score parameter or the data db was updated), pass `--rebuild-selected-table`.

### 3. generating metadata

```sh
python3 build_dataset_metadata.py data-$DATE.db
```

this will add metadata to the dataset (tags, post index).

## training the model

```sh
python3 train.py
```

then wait forever for it to finish.

## testing the model

```sh
python3 predict.py models/version_0/checkpoints/epoch=10-*.ckpt image.jpg
```

maybe the model works idk lol

## about the model

identifur uses a resnet-152 model pretrained against imagenet that is then trained against a tagged data set. each tag is a column in the label tensor.
