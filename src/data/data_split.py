import pandas as pd
from typing import List


def split1(df: pd.DataFrame) -> List[pd.DataFrame]:
    test_q = 0.85

    train_size = int(test_q * len(df) - (test_q * len(df) % 60))

    x_train, x_val = df[:train_size], df[train_size:]

    # X_train.shape, X_test.shape ((1324320, 13), (233760, 13))
    assert x_train.shape[0] == 1324320
    assert x_val.shape[0] == 233760

    return (x_train, x_val, None)


def split(df: pd.DataFrame) -> List[pd.DataFrame]:
    return split1(df)
