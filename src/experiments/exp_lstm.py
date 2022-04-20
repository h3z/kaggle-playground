from tensorflow import keras as k
import typing

if typing.TYPE_CHECKING:
    print("emmm")
    from keras.api._v2 import keras as k


def lstm_model():
    model = k.models.Sequential()
    model.add(k.layers.Input(shape=(60, 13)))
    model.add(k.layers.LSTM(32))
    model.add(k.layers.Dense(1, activation="sigmoid"))
    return model
