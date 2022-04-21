from tensorflow import keras as k
import typing

if typing.TYPE_CHECKING:
    print("emmm")
    from keras.api._v2 import keras as k


def lstm_model():

    input1 = k.layers.Input(shape=(60, 13))
    x1 = k.layers.LSTM(32)(input1)
    x1 = k.layers.BatchNormalization()(x1)
    x1 = k.layers.Dense(16, activation="relu")(x1)
    x1 = k.layers.BatchNormalization()(x1)

    x1 = k.layers.Dense(64, activation="relu")(x1)
    x1 = k.layers.BatchNormalization()(x1)

    input2 = k.layers.Input(shape=(1))
    # x2 = k.layers.Embedding(1, 16)(input2)
    x2 = k.layers.Dense(16, activation="relu")(input2)
    x2 = k.layers.BatchNormalization()(x2)
    x2 = k.layers.Dense(32, activation="relu")(x2)
    x2 = k.layers.BatchNormalization()(x2)
    # x2 = tf.squeeze(x2, axis=1)
    # x2 = input2

    x = k.layers.Concatenate(axis=-1)([x1, x2])
    x = k.layers.Dense(64, activation="relu")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Dense(128, activation="relu")(x)
    x = k.layers.BatchNormalization()(x)
    x = k.layers.Dense(16, activation="relu")(x)
    output = k.layers.Dense(1, activation="sigmoid")(x)

    model = k.Model(inputs=[input1, input2], outputs=output)
    return model
