import wandb
import tensorflow as tf

from tensorflow import keras as k
import typing

if typing.TYPE_CHECKING:
    from keras.api._v2 import keras as k


def get():

    if wandb.config.optimizer == "adam":
        # return k.optimizers.Adam(wandb.config.lr)
        return k.optimizers.Adam(1e-7)
