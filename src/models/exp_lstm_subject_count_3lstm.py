from tensorflow import keras as k
import typing

if typing.TYPE_CHECKING:
    print("emmm")
    from keras.api._v2 import keras as k


def lstm_model():

    input1 = k.layers.Input(shape=(60, 13))
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
