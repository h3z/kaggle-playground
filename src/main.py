import wandb, utils, os, sys, warnings
from model import models
from data import data_split, data_process, data_loader, data_reader
from train import train, losses, optimizers, schedulers
from callback import early_stopping, wandb_callback, callback
from config import config
from config.config import sensor_cols
from typing import List

# os.chdir("/home/yanhuize/kaggle/TPS-Apr/src")
warnings.filterwarnings("ignore")
utils.fix_random()
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_parameters():
    return {
        "~lr": 0.0001,
        "~batch_size": 128,
        "~epochs": 200,
        "~early_stopping_patience": 3,
        "~optimizer": "adam",
        "~loss": "mse",
    }


def main():
    wandb.init(config=get_parameters(), **config.__wandb__)
    print(wandb.config)

    # read csv
    data_util = data_reader.DataReader()
    train_df = data_util.train
    test_df = data_util.test
    test_df["state"] = -1  # 为了对齐

    # split
    train_df, val_df, _ = data_split.split(train_df)

    # preprocess
    processor = data_process.DataProcess(train_df[sensor_cols])
    train_df.loc[:, sensor_cols] = processor.preprocess(train_df[sensor_cols])
    val_df.loc[:, sensor_cols] = processor.preprocess(val_df[sensor_cols])
    test_df.loc[:, sensor_cols] = processor.preprocess(test_df[sensor_cols])

    # torch DataLoader
    train_ds = data_loader.DataLoader(train_df).get(is_train=True)
    val_ds = data_loader.DataLoader(val_df).get()
    test_ds = data_loader.DataLoader(test_df).get()

    model = models.get()

    # train
    criterion = losses.get()
    optimizer = optimizers.get(model)
    scheduler = schedulers.get(optimizer, train_ds)
    callbacks: List[callback.Callback] = [
        early_stopping.EarlyStopping(),
        wandb_callback.WandbCallback(),
    ]

    for epoch in range(wandb.config["~epochs"]):
        loss = train.epoch_train(
            model, optimizer, scheduler, train_ds, criterion, callbacks
        )
        val_loss = train.epoch_val(model, val_ds, criterion, callbacks)
        print(epoch, ": train_loss", loss, "val_loss", val_loss)

        res = [c.on_epoch_end(loss, val_loss, model) for c in callbacks]
        if False in res:
            print("Early stopping")
            break

    [c.on_train_finish(model) for c in callbacks]

    # predict
    preds, gts = train.predict(model, test_ds)

    # post process
    preds, gts = processor.postprocess(preds), processor.postprocess(gts)

    data_util.submit(preds)
    wandb.finish()


if __name__ == "__main__":
    main()
