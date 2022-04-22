import pandas as pd
from typing import List
import config as C
from sklearn.model_selection import GroupKFold
import wandb


def split(df: pd.DataFrame, label_df: pd.DataFrame) -> List[pd.DataFrame]:
    if wandb.config.split_type == 1:
        return split_1(df, label_df)
    elif wandb.config.split_type == 2:
        return split_2(df, label_df)


def split_1(df: pd.DataFrame, label_df: pd.DataFrame) -> List[pd.DataFrame]:
    df = df.pivot(index=["sequence", "subject"], columns="step", values=C.sensor_cols)
    train_idx, val_idx = next(
        GroupKFold(n_splits=5).split(df, groups=df.index.get_level_values(1))
    )

    x_train, y_train, x_val, y_val = (
        df.iloc[train_idx].stack().reset_index(),
        label_df.iloc[train_idx],
        df.iloc[val_idx].stack().reset_index(),
        label_df.iloc[val_idx],
    )

    return x_train, y_train, x_val, y_val


def split_2(df: pd.DataFrame, label_df: pd.DataFrame) -> List[pd.DataFrame]:
    test_q = 0.85

    train_size = int(test_q * len(df) - (test_q * len(df) % 60))
    train_label_size = int(test_q * len(label_df))

    x_train, y_train = df[:train_size], label_df[:train_label_size]
    x_val, y_val = df[train_size:], label_df[train_label_size:]

    # X_train.shape, X_test.shape ((1324320, 13), (233760, 13))
    assert x_train.shape[0] == 1324320
    assert x_val.shape[0] == 233760

    return x_train, y_train, x_val, y_val
