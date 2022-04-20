import numpy as np
import pandas as pd
import config as C
from sklearn.preprocessing import StandardScaler


class DataProcess:
    def __init__(self, train_df: pd.DataFrame) -> None:
        self.scaler = StandardScaler()
        self.scaler.fit_transform(train_df[C.sensor_cols])

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        df[C.sensor_cols] = self.scaler.transform(df[C.sensor_cols])

        # subject count feature
        df = pd.merge(
            df,
            df.subject.value_counts().rename("subject_count"),
            left_on="subject",
            right_index=True,
        )
        return df
