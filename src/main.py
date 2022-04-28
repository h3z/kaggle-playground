import wandb, utils, os, sys, warnings, multiprocessing, traceback
from model import models
from data import data_split, data_process, data_loader
from train import train, losses, optimizers, schedulers
from callback import early_stopping, wandb_callback, callback
from config import config
from config.config import sensor_cols
from typing import List
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


# os.chdir("/home/yanhuize/kaggle/TPS-Apr/src")
warnings.filterwarnings("ignore")
utils.fix_random()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# def reset_wandb_env():
#     exclude = {
#         "WANDB_PROJECT",
#         "WANDB_ENTITY",
#         "WANDB_API_KEY",
#     }
#     for k, v in os.environ.items():
#         if k.startswith("WANDB_") and k not in exclude:
#             del os.environ[k]


def run_group(params: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, **kwargs):
    print(kwargs)
    device = kwargs["device"]
    model_q: multiprocessing.Queue = kwargs["models"]
    wandb.init(config=params, **config.__wandb__)

    # torch DataLoader
    train_ds = data_loader.DataLoader(train_df).get(is_train=True)
    val_ds = data_loader.DataLoader(val_df).get()

    model = models.get(device)

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
            model, optimizer, scheduler, train_ds, criterion, callbacks, device=device
        )
        val_loss = train.epoch_val(model, val_ds, criterion, callbacks, device=device)
        print(epoch, ": train_loss", loss, "val_loss", val_loss)

        res = [c.on_epoch_end(loss, val_loss, model) for c in callbacks]
        if False in res:
            print("Early stopping")
            break

    [c.on_train_finish(model) for c in callbacks]

    wandb.finish()

    model_q.put(model)


def run_group_wrap(args, kwargs):
    try:
        return run_group(*args, **kwargs)
    except:
        traceback.print_exc()


def main():
    params = {
        "~lr": 0.0001,
        "~batch_size": 128,
        "~epochs": 200,
        "~early_stopping_patience": 10,
        "~optimizer": "adam",
        "~loss": "bce",  # bce, mse
        "layer": 6,
        "hidden_size": 256,
        "bidirectional": True,
        "group": "try_group",
    }
    print(params)

    data = data_process.DataFE()
    train_df, test_df = data.train, data.test
    train_df = train_df.pivot(
        index=["sequence", "subject", "state"],
        columns="step",
        values=config.sensor_cols,
    )

    models_q = multiprocessing.Queue()
    num_folds = 5
    kfd = KFold(n_splits=num_folds, shuffle=True)
    for group_n, (train_idx, val_idx) in enumerate(kfd.split(train_df)):
        multiprocessing.Process(
            target=run_group_wrap,
            kwargs=dict(
                args=[
                    params,
                    train_df.iloc[train_idx].stack().reset_index(),
                    train_df.iloc[val_idx].stack().reset_index(),
                ],
                kwargs={
                    "group_n": group_n,
                    "models": models_q,
                    "device": f"cuda:{group_n}",
                },
            ),
        ).start()

    models = [models_q.get() for i in range(num_folds)]

    # predict
    test_ds = data_loader.DataLoader(test_df).get()
    preds = [train.predict(model, test_ds)[0].reshape(-1) for model in models]

    # submit
    data.submit(np.array(preds).mean(0))


if __name__ == "__main__":
    main()
