import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from config import config
import wandb, utils
from config.config import DATA_PATH


class DataFE:
    def __init__(self) -> None:
        self.submission = pd.read_csv(f"{DATA_PATH}/csv/sample_submission.csv")
        self.train = pd.read_pickle(f"{DATA_PATH}/pkl/train.pkl")
        self.test = pd.read_pickle(f"{DATA_PATH}/pkl/test.pkl")
        self.label = pd.read_csv(f"{DATA_PATH}/csv/train_labels.csv")

        # concat state column
        self.train = self.train.merge(self.label, on="sequence")
        self.test["state"] = -1

        # fillna
        self.train = self.train.fillna(0)
        self.test = self.test.fillna(0)

        # scaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.train[config.sensor_cols].values)

        self.preprocess()

    def preprocess(self):
        sensor_cols = config.sensor_cols
        self.train[sensor_cols] = self.scaler.transform(self.train[sensor_cols].values)
        self.test[sensor_cols] = self.scaler.transform(self.test[sensor_cols].values)

    def postprocess(self, arr: np.ndarray) -> np.ndarray:
        return arr

    def submit(self, preds: np.ndarray):
        self.submission["state"] = preds.squeeze()

        f = utils.mktemp("submit.csv")
        self.submission.to_csv(f, index=False)
        # wandb.save(f)
        print(f)
