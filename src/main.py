from numpy import fliplr
from Experiment import Experiment
from experiments.exp_lstm_subject import exp_lstm_subject
from experiments.exp_lstm import exp_lstm

from Dataset import Dataset
import config as C
import wandb
import os
from typing import Type
import glob
from pathlib import Path


def get_parameters():
    return {"lr": 0.001, "epochs": 200, "batch_size": 128}


def run_exp(EXP: Type[Experiment]):
    wandb.init(project="Tabular-Playground-Series-Apr-2022", entity="hzzz")
    wandb.config.update(get_parameters())

    ds = Dataset()
    exp = EXP(ds, wandb.config)

    final_result = exp.train()
    wandb.log(final_result)

    preds = exp.predict()

    file_path = ds.submit_result(preds)

    for i in C.PROJECT_PATH.glob("**/*.py"):
        if "wandb" in str(i):
            continue
        wandb.save(
            str(i),
            C.PROJECT_PATH,
            policy="now",
        )
    wandb.save(file_path, Path(file_path).parent, "now")
    wandb.finish()


if __name__ == "__main__":
    os.environ["http_proxy"] = "http://localhost:6152"
    os.environ["https_proxy"] = "http://localhost:6152"
    os.environ["NEPTUNE_ALLOW_SELF_SIGNED_CERTIFICATE"] = "TRUE"

    run_exp(exp_lstm)
    # run_exp(exp_lstm_subject)
