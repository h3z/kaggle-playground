import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class DataProcess:
    def __init__(self, df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.scaler.fit(df.fillna(0).values)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.scaler.transform(df.fillna(0).values)

    def postprocess(self, arr: np.ndarray) -> np.ndarray:
        # return self.scaler.inverse_transform(preds)
        return arr
