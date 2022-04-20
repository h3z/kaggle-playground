from pathlib import Path

PROJECT_PATH = Path("/home/yanhuize/kaggle/Tabular-Playground-Series-Apr-2022")
DATASET_PATH = PROJECT_PATH / "dataset"


sensor_cols = [f"sensor{i:02d}" for i in range(13)]
