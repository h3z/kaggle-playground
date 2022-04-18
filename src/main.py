from base_experiment import Experiment

from experiments.exp_lstm_subject import EXP as exp

from experiments.exp_lstm_subject_count import EXP as exp2

from dataset_util import Dataset
import config as C
import wandb
from typing import Type
import random
import tensorflow as tf

random.seed(42)
tf.random.set_seed(42)


def get_parameters():
    return {"lr": 0.001, "epochs": 50, "batch_size": 256, "exp": 1}


def run_exp(EXP: Type[Experiment]):

    ds = Dataset()
    exp = EXP(ds, wandb.config)

    final_result = exp.train()
    wandb.log(final_result)

    preds = exp.predict()
    ds.submit_result(preds)

    wandb.finish()


if __name__ == "__main__":
    wandb.init(project="Tabular-Playground-Series-Apr-2022", entity="hzzz")
    wandb.config.update(get_parameters())
    if wandb.config.exp == 1:
        run_exp(exp)
    elif wandb.config.exp == 2:
        run_exp(exp2)
