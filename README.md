# identifur

identifur tries to automatically tag furry art.

## Fetching data

identifur has a bunch of scripts you should use to fetch the dataset. the scripts fetch data from e621, so you're gonna end up with a lot of furry porn.

### 1. Fetch the data DB

```sh
python3 fetch_e621_data_db.py $DATE
```

this generates a `data-$DATE.db` file for use with everything else. if you're not sure what to use for the date, check the files in https://e621.net/db_export/.

### 2. Generate the downloads DB

```sh
python3 make_dls_db.py data-$DATE.db
```

this generates a `dls.db` file that is used to keep track of downloads. by default, posts with tags that depict underage content will be blacklisted because that's fucking gross.

### 3. Fetch the dataset

```sh
python3 fetch_e621_dataset.py data-$DATE.db
```

this will start populating the `dataset` directory. you can interrupt and restart the script at any time: current status is saved to `dls.db`.

## Training the model

```sh
python3 train.py data-$DATE.db
```

then wait forever for it to finish.

## Testing the model

```sh
python3 test.py data-$DATE.db models/best/$MODEL.pt image.jpg
```

maybe the model works idk lol
