# identifur

identifur tries to automatically tag furry art.

## fetching the dataset

identifur has a bunch of scripts you should use to fetch the dataset. the scripts fetch data from e621, so you're gonna end up with a lot of furry porn.

### 1. fetch the data DB

```sh
python3 fetch_e621_data_db.py $DATE
```

this generates a `data-$DATE.db` file for use with everything else. if you're not sure what to use for the date, check the files in https://e621.net/db_export/.

### 2. fetch the images

```sh
python3 fetch_e621_images.py data-$DATE.db
```

this will start populating the `images` directory. you can interrupt and restart the script at any time: current status is saved to `dls.db`. if you need to update the selections (e.g. due to a different minimum score parameter or the data db was updated), pass `--rebuild-selected-table`.

### 3. build a huggingface dataset

```sh
python3 build_huggingface_dataset.py data-$DATE.db
```

this will build a huggingface dataset for use with training.

## training the model

```sh
python3 train.py
```

then wait forever for it to finish.

## testing the model

```sh
python3 predict.py models/version_0/checkpoints/epoch=10-*.ckpt < image.jpg
```

or the really bad web ui:

```sh
python3 -m identifur.web models/version_0/checkpoints/epoch=10-*.ckpt
```
