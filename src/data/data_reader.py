import pandas as pd

from config.config import DATA_PATH
import numpy as np


class DataReader:
    def __init__(self):
        self.train = pd.read_pickle(f"{DATA_PATH}/pkl/train.pkl")
        self.test = pd.read_pickle(f"{DATA_PATH}/pkl/test.pkl")
        self.label = pd.read_csv(f"{DATA_PATH}/csv/train_labels.csv")

        self.train = self.train.merge(self.label, on="sequence")
        self.submission = pd.read_csv(f"{DATA_PATH}/csv/sample_submission.csv")

    def submit(self, preds: np.ndarray):
        self.submission["state"] = preds.squeeze()
        self.submission.to_csv("submit.csv", index=False)
