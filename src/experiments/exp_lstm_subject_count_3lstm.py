from lib2to3.pgen2.pgen import DFAState
from base_experiment import Experiment
from dataset_util import Dataset
import tensorflow as tf

from tensorflow import keras as k
import typing
import config as C
import wandb
import pandas as pd
from sklearn.model_selection import GroupKFold

if typing.TYPE_CHECKING:
    print("emmm")
    from keras.api._v2 import keras as k


def lstm_model(timesteps, feature):

    input1 = k.layers.Input(shape=(timesteps, feature))
    x1 = k.layers.Bidirectional(k.layers.LSTM(512, return_sequences=True))(input1)
    x1 = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=True))(x1)
    x1 = k.layers.Bidirectional(k.layers.LSTM(256, return_sequences=False))(x1)
    x1 = k.layers.Dense(256, activation="relu")(x1)

    input2 = k.layers.Input(shape=(1))
    x2 = k.layers.Dense(32, activation="relu")(input2)
    x2 = k.layers.Dense(64, activation="relu")(x2)

    x = k.layers.Concatenate(axis=-1)([x1, x2])
    x = k.layers.Dense(256, activation="relu")(x)
    output = k.layers.Dense(1, activation="sigmoid")(x)

    model = k.Model(inputs=[input1, input2], outputs=output)
    return model


class EXP(Experiment):
    def __init__(self, ds: Dataset, params) -> None:
        super().__init__(ds, params)
        self.model = lstm_model(ds.train.step.nunique(), len(ds.sensor_cols))

    def train(self):
        self.model.compile(
            optimizer=k.optimizers.Adam(learning_rate=self.params["lr"]),
            loss=k.losses.BinaryCrossentropy(),
            metrics=[k.metrics.BinaryAccuracy()],
        )

        train_xy, val_xy = self.get_dataset()

        history = self.model.fit(
            train_xy,
            validation_data=val_xy,
            epochs=self.params["epochs"],
            callbacks=[wandb.keras.WandbCallback()],
        )

        final_result = {}
        for metric in history.history.keys():
            final_result[metric] = history.history[metric][-1]
        return final_result

    def predict(self):
        x = self.preprocess(self.ds.test)
        preds = self.model.predict(x)
        return preds

    def get_dataset(self):
        df = self.ds.train

        def dataloader(idx, type):
            temp = df.iloc[idx]
            return (
                tf.data.Dataset.from_tensor_slices(self.preprocess(temp, type))
                .batch(self.params["batch_size"])
                .shuffle(self.params["batch_size"] * 4)
            )

        train_idx, val_idx = next(GroupKFold(n_splits=5).split(df, groups=df.subject))
        return dataloader(train_idx, "train"), dataloader(val_idx, "val")

    def preprocess(self, df, type="test"):
        temp = pd.merge(
            df,
            df.subject.value_counts().rename("subject_count"),
            left_on="subject",
            right_index=True,
        )

        def f(cols):
            return temp[cols].values.reshape(-1, self.ds.timesteps, len(cols))

        if type != "test":
            temp = pd.merge(temp, self.ds.label, on="sequence")
            return (f(self.ds.sensor_cols), f(["subject_count"])[:, 0]), f(["state"])[
                :, 0
            ]

        else:
            return (f(self.ds.sensor_cols), f(["subject_count"])[:, 0])

    @property
    def name(self):
        return "lstm with subject"

    @property
    def description(self):
        return "lstm with subject (multi-input)"
