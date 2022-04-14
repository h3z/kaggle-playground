from pathlib import Path
import neptune.new as neptune
import configparser
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

PROJECT_PATH = Path("/home/yanhuize/kaggle/Tabular-Playground-Series-Apr-2022")
DATASET_PATH = PROJECT_PATH / "dataset"


config = configparser.ConfigParser()
config.read(DATASET_PATH / "neptune.ini")
NEPTUNE_AI_TOKEN = config["A"]["token"]

import wandb
from wandb.keras import WandbCallback


WANDB = "WANDB"
NEPTUNE = "NEPTUNE"


class ML_TOOL:
    def __init__(self, type, params) -> None:
        self.type = type
        self.params = params
        if self.type == WANDB:
            self.run = wandb.init(
                project="Tabular-Playground-Series-Apr-2022", entity="hzzz"
            )
        elif self.type == NEPTUNE:
            run = neptune.init(
                project="h3z/Tabular-Playground-Series-Apr-2022",
                api_token=NEPTUNE_AI_TOKEN,
                proxies={
                    "http": "http://localhost:6152",
                    "https": "http://localhost:6152",
                },
                source_files="**/*.py",
            )
            self.run = run

    def callback(self):
        if self.type == WANDB:
            return WandbCallback()
        elif self.type == NEPTUNE:
            return NeptuneCallback(run=self.run, base_namespace="training")

    def record(self, exp, result, submit_file):
        if self.type == NEPTUNE:
            self.run["sys/name"] = exp.name
            self.run["sys/description"] = exp.description

            for k, v in result.items():
                self.run[f"eval/{k}"] = v
            self.run["params"] = self.params
            self.run["submit"].upload(submit_file)
        elif self.type == WANDB:
            wandb.log(result)
            wandb.log({"params": self.params})
            # wandb.log({"submit": submit_file})

    def stop(self):
        if self.type == NEPTUNE:
            self.run.stop()
        elif self.type == WANDB:
            self.run.save("**/*.py")
            self.run.finish()
