import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import IPython.display as ipd
from itertools import cycle
from typing import TYPE_CHECKING
import config as C


class Dataset:
    def __init__(self) -> None:
        self.train = self.rename_columns(
            pd.read_pickle(f"{C.DATASET_PATH}/pkl/train.pkl")
        )
        self.test = self.rename_columns(
            pd.read_pickle(f"{C.DATASET_PATH}/pkl/test.pkl")
        )
        self.label = pd.read_csv(f"{C.DATASET_PATH}/csv/train_labels.csv")

    def submit_result(self, pred):
        submit = pd.read_csv(f"{C.DATASET_PATH}/csv/sample_submission.csv")
        submit.iloc[:, 1] = pred
        file_path = f"{C.DATASET_PATH}/submit.csv"
        submit.to_csv(file_path, index=False)
        return file_path

    def rename_columns(self, df):

        df.columns = [
            "sequence",
            "subject",
            "step",
            "sensor00",
            "sensor01",
            "sensor02",
            "sensor03",
            "sensor04",
            "sensor05",
            "sensor06",
            "sensor07",
            "sensor08",
            "sensor09",
            "sensor10",
            "sensor11",
            "sensor12",
        ]

        return df

    @property
    def sensor_cols(self):
        return [f"sensor{i:02d}" for i in range(13)]

    @property
    def timesteps(self):
        return self.train.step.nunique()
