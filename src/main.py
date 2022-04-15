from base_experiment import Experiment
from experiments.exp_lstm_subject import exp_lstm_subject

from dataset_util import Dataset
import config as C
import wandb
from typing import Type


def get_parameters():
    return {"lr": 0.001, "epochs": 30, "batch_size": 128}


def run_exp(EXP: Type[Experiment]):
    wandb.init(project="Tabular-Playground-Series-Apr-2022", entity="hzzz")
    wandb.config.update(get_parameters())

    ds = Dataset()
    exp = EXP(ds, wandb.config)

    final_result = exp.train()
    wandb.log(final_result)

    preds = exp.predict()

    file_path = ds.submit_result(preds)
    wandb.finish()


if __name__ == "__main__":
    run_exp(exp_lstm_subject)
