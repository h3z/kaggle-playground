from base_experiment import Experiment
from dataset_util import Dataset
import tensorflow as tf

from tensorflow import keras as k
import typing
import config as C
import wandb

if typing.TYPE_CHECKING:
    print("emmm")
    from keras.api._v2 import keras as k


def lstm_model(timesteps, feature):
    model = k.models.Sequential()
    model.add(k.layers.Input(shape=(timesteps, feature)))
    model.add(k.layers.LSTM(32))
    model.add(k.layers.Dense(1, activation="sigmoid"))
    return model


class exp_lstm(Experiment):
    def __init__(self, ds: Dataset, params) -> None:
        super().__init__(ds, params)
        self.model = lstm_model(ds.train.step.nunique(), len(ds.sensor_cols))

    def train(self):
        self.model.compile(
            optimizer=k.optimizers.Adam(),
            loss=k.losses.BinaryCrossentropy(),
            metrics=[k.metrics.BinaryAccuracy()],
        )

        xy = self.get_dataset()
        L = int(len(xy) * 0.8)
        history = self.model.fit(
            xy.take(L),
            validation_data=xy.skip(L),
            epochs=self.params["epochs"],
            callbacks=[wandb.keras.WandbCallback()],
        )

        final_result = {}
        for metric in history.history.keys():
            final_result[metric] = history.history[metric][-1]
        return final_result

    def predict(self):
        x = self.preprocess_x(self.ds.test)
        preds = self.model.predict(x)
        return preds

    def get_dataset(self):
        ds = self.ds
        return (
            tf.data.Dataset.from_tensor_slices(
                (
                    self.preprocess_x(ds.train),
                    ds.label.state.values,
                )
            )
            .batch(self.params["batch_size"])
            .shuffle(self.params["batch_size"] * 4)
        )

    def preprocess_x(self, df):
        return df[self.ds.sensor_cols].values.reshape(-1, self.ds.timesteps, 13)

    @property
    def name(self):
        return "base lstm"

    @property
    def description(self):
        return "base lstm, only with timeseries sensor data"
