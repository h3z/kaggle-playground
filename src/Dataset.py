import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import seaborn as sns
from glob import glob
import IPython.display as ipd
from itertools import cycle
from typing import TYPE_CHECKING


class Dataset:
    def __init__(self) -> None:
        self._data_path = "../dataset/"

        self.train = self.rename_columns(
            pd.read_pickle(f"{self._data_path}/pkl/train.pkl")
        )
        self.test = self.rename_columns(
            pd.read_pickle(f"{self._data_path}/pkl/test.pkl")
        )
        self.label = pd.read_csv(f"{self._data_path}/csv/train_labels.csv")

    def submit_result(self, pred):
        submit = pd.read_csv(f"{self._data_path}/csv/sample_submission.csv")
        submit.iloc[:, 1] = pred
        submit.to_csv(f"{self._data_path}/submit.csv", index=False)

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
