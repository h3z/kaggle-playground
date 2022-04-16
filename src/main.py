import torch
from base_experiment import Experiment
from experiments.exp_lstm_subject import exp_lstm_subject

from dataset_util import Dataset
import config as C
import wandb
from typing import Type
import random
import tensorflow as tf

random.seed(42)
tf.random.set_seed(42)


def get_parameters():
    return {"lr": 0.001, "epochs": 100, "batch_size": 64}


def run_exp(EXP: Type[Experiment]):
    wandb.init(project="Tabular-Playground-Series-Apr-2022", entity="hzzz")
    wandb.config.update(get_parameters())

    ds = Dataset()
    exp = EXP(ds, wandb.config)

    final_result = exp.train()
    wandb.log(final_result)

    preds = exp.predict()
    ds.submit_result(preds)

    wandb.finish()


if __name__ == "__main__":
    run_exp(exp_lstm_subject)
