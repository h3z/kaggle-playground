from Experiment import Experiment
from Dataset import Dataset
import tensorflow as tf

from tensorflow import keras as k
import typing
import config as C

if typing.TYPE_CHECKING:
    print("emmm")
    from keras.api._v2 import keras as k


def lstm_model(timesteps, feature):

    input1 = k.layers.Input(shape=(timesteps, feature))
    x1 = k.layers.LSTM(32)(input1)
    x1 = k.layers.Dense(16)(x1)

    input2 = k.layers.Input(shape=(1))
    x2 = k.layers.Embedding(1, 16)(input2)
    x2 = tf.squeeze(x2, axis=1)

    x = k.layers.Concatenate(axis=-1)([x1, x2])
    x = k.layers.Dense(16)(x)
    output = k.layers.Dense(1, activation="sigmoid")(x)

    model = k.Model(inputs=[input1, input2], outputs=output)
    return model


class exp_lstm_subject(Experiment):
    def __init__(self, run, ds: Dataset, params) -> None:
        super().__init__(run, ds, params)
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
            callbacks=[C.neptune_callback(self.run)],
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
                    ds.label.state.values.reshape(-1, 1),
                )
            )
            .batch(self.params["batch_size"])
            .shuffle(self.params["batch_size"] * 4)
        )

    def preprocess_x(self, df):
        ds = self.ds

        x = (
            # x: time series
            df[ds.sensor_cols].values.reshape(-1, ds.timesteps, 13),
            # x: subject
            df.groupby("sequence").subject.first().values.reshape(-1, 1),
        )

        return x

    @property
    def name(self):
        return "lstm with subject"

    @property
    def description(self):
        return "lstm with subject (multi-input)"
