import wandb
from data import data1, data2, data3


def get():
    if wandb.config.data == 1:
        return data1.Data()
    elif wandb.config.data == 2:
        return data2.Data()
    elif wandb.config.data == 3:
        return data3.Data()
