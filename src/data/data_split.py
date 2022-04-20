import pandas as pd
from typing import List
import config as C
from sklearn.model_selection import GroupKFold


class DataSplit:
    def __init__(self) -> None:
        pass

    def split(self, df: pd.DataFrame, label_df: pd.DataFrame) -> List[pd.DataFrame]:
        df = df.pivot(
            index=["sequence", "subject"], columns="step", values=C.sensor_cols
        )
        train_idx, val_idx = next(
            GroupKFold(n_splits=5).split(df, groups=df.index.get_level_values(1))
        )

        return (
            df.iloc[train_idx].stack().reset_index(),
            df.iloc[val_idx].stack().reset_index(),
            label_df.iloc[train_idx],
            label_df.iloc[val_idx],
        )
