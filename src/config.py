from pathlib import Path
import neptune.new as neptune
import configparser
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

PROJECT_PATH = Path("/home/yanhuize/kaggle/Tabular-Playground-Series-Apr-2022")
DATASET_PATH = PROJECT_PATH / "dataset"


config = configparser.ConfigParser()
config.read(DATASET_PATH / "neptune.ini")
NEPTUNE_AI_TOKEN = config["A"]["token"]


def init_neptune():
    run = neptune.init(
        project="h3z/Tabular-Playground-Series-Apr-2022",
        # name=name,
        # description=description,
        # run="TAB-11",
        api_token=NEPTUNE_AI_TOKEN,
        proxies={
            "http": "http://localhost:6152",
            "https": "http://localhost:6152",
        },
        source_files="**/*.py",
    )
    return run


def neptune_callback(run):
    return NeptuneCallback(run=run, base_namespace="training")
