PROJECT_NAME = "TPS-Apr-2022"
ONLINE = True

__wandb__ = {
    "project": PROJECT_NAME,
    "entity": "hzzz",
    "dir": f"/wandb/{PROJECT_NAME}",
    "mode": "online" if ONLINE else "offline",
}


DATA_PATH = "/home/yanhuize/kaggle/TPS-Apr/dataset"

sensor_cols = [f"sensor_{i:02d}" for i in range(13)]
