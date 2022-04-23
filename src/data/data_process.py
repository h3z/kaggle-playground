import numpy as np
import pandas as pd
import config as C
from sklearn.preprocessing import StandardScaler


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print("%s cost time: %.3f s" % (func.__name__, time_spend))
        return result

    return func_wrapper


class DataProcess:
    def __init__(self, train_df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.scaler.fit_transform(train_df[C.sensor_cols])

    @timer
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[C.sensor_cols] = self.scaler.transform(df[C.sensor_cols])

        # subject count feature
        df = pd.merge(
            df,
            df.subject.value_counts().rename("subject_count"),
            left_on="subject",
            right_index=True,
        )
        return df
