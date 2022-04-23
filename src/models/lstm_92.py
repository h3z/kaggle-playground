from tensorflow import keras as k

import tensorflow as tf
import typing

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras as k


Bidirectional = k.layers.Bidirectional
LSTM = k.layers.LSTM


def get_model():
    model = k.Sequential(
        [
            # lstm
            Bidirectional(LSTM(512, return_sequences=True)),
            Bidirectional(LSTM(512, return_sequences=True)),
            # lstm1
            Bidirectional(LSTM(512, return_sequences=True)),
            Bidirectional(LSTM(512, return_sequences=True)),
            # lstm2
            Bidirectional(LSTM(512, return_sequences=True)),
            Bidirectional(LSTM(512, return_sequences=True)),
            k.layers.Reshape((-1,)),
            k.layers.ReLU(),
            k.layers.Dense(1),
            # k.layers.Dense(1, activation="sigmoid"),
        ]
    )
    # model.build(input_shape=(32, 60, 13))
    return model
