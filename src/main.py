# from experiments import exp_lstm as exp0
from experiments import exp_lstm_subject_count_3lstm_scaler as exp0

# from experiments.exp_lstm_subject import EXP as exp
# from experiments.exp_lstm_subject_count import EXP as exp2
# from experiments.exp_lstm_subject_count_3lstm import EXP as exp3

from data.data_reader import DataReader
import wandb
import random
import tensorflow as tf
from data.data_process import DataProcess
from data import data1, data2
from tensorflow import keras as k
import typing
from data.data_split import DataSplit
import os

os.environ["WANDB_MODE"] = "offline"
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[0], "GPU")

if typing.TYPE_CHECKING:
    print("emmm")
    from keras.api._v2 import keras as k

random.seed(42)
tf.random.set_seed(42)


def get_parameters():
    return {"lr": 0.0001, "epochs": 4, "batch_size": 1024}


def main():

    # read data
    data_util = DataReader()
    train_df = data_util.train
    test_df = data_util.test
    label_df = data_util.label

    # split data
    data_split = DataSplit()
    train_x_df, val_x_df, train_y_df, val_y_df = data_split.split(train_df, label_df)

    # preprocess
    data_process = DataProcess(train_x_df)
    train_x_df = data_process.preprocess(train_x_df)
    val_x_df = data_process.preprocess(val_x_df)
    test_x_df = data_process.preprocess(test_df)

    # tf.Dataset
    ds = data2.Data()
    train_ds = ds.get_train_ds(train_x_df, train_y_df)
    val_ds = ds.get_train_ds(val_x_df, val_y_df)

    # model
    model = exp0.lstm_model()

    # train
    model.compile(
        optimizer=k.optimizers.Adam(wandb.config.lr),
        loss=k.losses.BinaryCrossentropy(),
        metrics=[k.metrics.BinaryAccuracy()],
    )

    L = int(len(train_ds) * 0.8)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=wandb.config.epochs,
        callbacks=[wandb.keras.WandbCallback()],
    )

    # result
    for metric in history.history.keys():
        wandb.log({metric: history.history[metric][-1]})

    # eval
    test_ds = ds.get_test_ds(test_x_df)
    preds = model.predict(test_ds)
    data_util.submit_result(preds)


if __name__ == "__main__":
    wandb.init(project="TPS-Apr-2022", entity="hzzz", config=get_parameters())
    print(wandb.config)
    main()
    wandb.finish()
