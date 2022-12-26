#!/usr/bin/env python3
import argparse
import torch
import logging
from identifur import models
from identifur.data import E621Dataset
from torch import optim, nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.metrics import Accuracy, Loss
from ignite.handlers import FastaiLRFinder, Checkpoint, global_step_from_engine
from ignite.contrib.handlers.tqdm_logger import ProgressBar

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    filename="train.log",
    encoding="utf-8",
    level=logging.INFO,
)


class TransformingDataset(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        item = self.ds[index]
        if item is None:
            return None
        image, label = item
        return self.transform(image), label


def collate_fn(batch):
    return default_collate([item for item in batch if item is not None])


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("data_db")
    argparser.add_argument("--dataset-path", default="dataset")
    argparser.add_argument("--random-split-seed", default=42, type=int)
    argparser.add_argument("--tag-min-post-count", default=2500, type=int)
    argparser.add_argument("--learning-rate", default=None, type=float)
    argparser.add_argument("--trainer-log-interval", default=100, type=int)
    argparser.add_argument(
        "--trainer-iteration-checkpoint-intervals", default=1000, type=int
    )
    argparser.add_argument("--max-epochs", default=10, type=int)
    argparser.add_argument("--train-data-split", default=0.6, type=float)
    argparser.add_argument("--validation-data-split", default=0.2, type=float)
    argparser.add_argument("--test-data-split", default=0.2, type=float)
    args = argparser.parse_args()

    ds = E621Dataset(args.data_db, args.dataset_path, args.tag_min_post_count)

    logging.info("loaded dataset with %d tags", len(ds.tags))

    device = torch.device("cuda")

    model = models.model(
        pretrained=True, requires_grad=False, out_features=len(ds.tags)
    ).to(device)

    batch_size = 64
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001 if args.learning_rate is None else args.learning_rate,
    )
    criterion = nn.BCEWithLogitsLoss()

    train_data, val_data, test_data = random_split(
        ds,
        [args.train_data_split, args.validation_data_split, args.test_data_split],
        generator=torch.Generator().manual_seed(args.random_split_seed),
    )

    train_loader = DataLoader(
        TransformingDataset(
            train_data,
            transforms.Compose(
                [
                    transforms.Resize((400, 400)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=45),
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TransformingDataset(
            val_data,
            transforms.Compose(
                [
                    transforms.Resize((400, 400)),
                    transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    to_save = {"model": model, "optimizer": optimizer}

    trainer = create_supervised_trainer(model, optimizer, criterion, device=device)
    ProgressBar().attach(trainer)

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=args.trainer_iteration_checkpoint_intervals),
        Checkpoint(to_save, "models/current_iteration", n_saved=2),
    )
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        Checkpoint(
            to_save,
            "models/epoch",
            n_saved=2,
            global_step_transform=lambda *_: trainer.state.epoch,
        ),
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=args.trainer_log_interval))
    def log_training_loss(trainer):
        logging.info(
            "Epoch[%d], Iter[%d] Loss: %.2f",
            trainer.state.epoch,
            trainer.state.iteration,
            trainer.state.output,
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        logging.info(
            "Training Results - Epoch[%d] Avg accuracy: %.2f Avg loss: %.2f",
            trainer.state.epoch,
            metrics["accuracy"],
            metrics["loss"],
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        logging.info(
            "Validation Results - Epoch[%d] Avg accuracy: %.2f Avg loss: %.2f",
            trainer.state.epoch,
            metrics["accuracy"],
            metrics["loss"],
        )

    def thresholded_output_transform(output):
        y_pred, y = output
        y_pred = torch.sigmoid(y_pred)
        y_pred = (y_pred > 0.55).float()
        return y_pred, y

    val_metrics = {
        "accuracy": Accuracy(
            is_multilabel=True, output_transform=thresholded_output_transform
        ),
        "loss": Loss(criterion),
    }

    train_evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device
    )
    ProgressBar().attach(train_evaluator)

    val_evaluator = create_supervised_evaluator(
        model, metrics=val_metrics, device=device
    )
    ProgressBar().attach(val_evaluator)
    val_evaluator.add_event_handler(
        Events.COMPLETED,
        Checkpoint(
            to_save,
            "models/best",
            n_saved=2,
            filename_prefix="best",
            score_name="accuracy",
            global_step_transform=global_step_from_engine(trainer),
        ),
    )

    if args.learning_rate is None:
        logging.info("no learning rate specified, will try to find one")
        lr_finder = FastaiLRFinder()
        with lr_finder.attach(trainer, to_save, end_lr=1e-02) as trainer_with_lr_finder:
            trainer_with_lr_finder.run(train_loader)
        logging.info(
            "LR finder logs: %s, suggested LR: %.8f",
            lr_finder.get_results(),
            lr_finder.lr_suggestion(),
        )
        lr_finder.apply_suggested_lr(optimizer)

    trainer.run(train_loader, max_epochs=args.max_epochs)


if __name__ == "__main__":
    main()
