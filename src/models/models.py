import wandb

from models import lstm_92, exp_lstm_subject


def get():
    if wandb.config.model == "92":
        return lstm_92.get_model()
    elif wandb.config.model == "old":
        return exp_lstm_subject.lstm_model()
