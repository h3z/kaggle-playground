import numpy as np
import torch
import wandb

from callback.callback import Callback


class EarlyStopping(Callback):
    def __init__(self) -> None:
        self.patience = wandb.config["~early_stopping_patience"]
        self.min_loss = np.inf
        self.counter = 0
        self.best_state_dict = None

    def on_val_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_epoch_end(self, loss, val_loss, model: torch.nn.Module) -> bool:
        if val_loss < self.min_loss:
            self.min_loss = val_loss
            self.counter = 0
            self.best_state_dict = model.state_dict()
        else:
            self.counter += 1

        return self.counter < self.patience

    def on_train_finish(self, model: torch.nn.Module):
        torch.save(self.best_state_dict, "./test.h5")
        model.load_state_dict(self.best_state_dict)
