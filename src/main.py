# from experiments import exp_lstm as exp0
# from models import exp_lstm_subject_count_3lstm as exp0
from multiprocessing import shared_memory
from models import models

# from experiments.exp_lstm_subject import EXP as exp
# from experiments.exp_lstm_subject_count import EXP as exp2
# from experiments.exp_lstm_subject_count_3lstm import EXP as exp3


import wandb
import random
import tensorflow as tf
from data import datas, data_reader, data_split
from data.data_process import DataProcess
from tensorflow import keras as k
import typing
import os
from train import optimizers

os.environ["WANDB_MODE"] = "offline"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[0], "GPU")

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras as k

random.seed(42)
tf.random.set_seed(42)


def get_parameters():
    return {
        "lr": 0.0001,
        "epochs": 200,
        # "batch_size": 64,
        "batch_size": 512,
        "optimizer": "adam",
        "split_type": 2,
        # 92, old
        "model": "92",
        # 1 原数据
        # 2 加入 subject_count
        # 3 加如 subject
        "data": 1,
        "warmup": 0.1,
    }


def scheduler(epoch, lr):
    spd1 = wandb.config.lr / (wandb.config.epochs * wandb.config.warmup)
    spd2 = -wandb.config.lr / (wandb.config.epochs * (1 - wandb.config.warmup))
    if epoch <= wandb.config.epochs * wandb.config.warmup:
        return lr + spd1
    else:
        return lr + spd2


def main():
    wandb.init(project="TPS-Apr-2022", entity="hzzz", config=get_parameters())
    print(wandb.config)

    # read data
    data_util = data_reader.DataReader()
    train_df = data_util.train
    test_df = data_util.test
    label_df = data_util.label

    # split data
    train_x_df, train_y_df, val_x_df, val_y_df = data_split.split(train_df, label_df)

    # preprocess
    data_process = DataProcess(train_x_df)
    train_x_df = data_process.preprocess(train_x_df)
    val_x_df = data_process.preprocess(val_x_df)
    test_x_df = data_process.preprocess(test_df)

    # tf.Dataset
    ds = datas.get()
    train_ds = ds.get_train_ds(train_x_df, train_y_df)
    val_ds = ds.get_train_ds(val_x_df, val_y_df)

    # model
    # model = exp0.lstm_model()
    model = models.get()
    # model.summary()

    # train
    model.compile(
        optimizer=optimizers.get(),
        # loss=k.losses.BinaryCrossentropy(from_logits=True),
        loss=k.losses.MeanSquaredError(),
        metrics=[k.metrics.BinaryAccuracy()],
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=wandb.config.epochs,
        callbacks=[
            wandb.keras.WandbCallback(),
            k.callbacks.LearningRateScheduler(scheduler),
        ],
    )

    # result
    for metric in history.history.keys():
        wandb.log({metric: history.history[metric][-1]})

    # eval
    test_ds = ds.get_test_ds(test_x_df)
    preds = model.predict(test_ds)
    data_util.submit_result(preds)

    wandb.finish()


if __name__ == "__main__":
    main()
