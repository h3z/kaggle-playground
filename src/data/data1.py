import pandas as pd
import tensorflow as tf
import config as C
import numpy as np
import wandb


class Data:
    def __init__(self) -> None:
        pass

    def get_train_ds(self, x, y):
        x = self.process_x(x)
        y = y.state.values
        return (
            tf.data.Dataset.from_tensor_slices((x, y)).batch(wandb.config.batch_size)
            # .shuffle(wandb.config.batch_size * 4)
        )

    def get_test_ds(self, x):
        x = self.process_x(x)
        return x

    def process_x(self, df: pd.DataFrame) -> np.ndarray:
        return df[C.sensor_cols].values.reshape(-1, 60, 13)
