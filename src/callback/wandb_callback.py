import numpy as np
import torch
import wandb

from callback.callback import Callback


class WandbCallback(Callback):
    def __init__(self) -> None:
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.train_batch_losses = []
        self.val_batch_losses = []

    def on_val_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        pass

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        wandb.log({"max_pred": preds.max(), "max_gt": gts.max()})
        self.train_batch_losses.append(loss)

    def on_epoch_end(self, loss, val_loss, model: torch.nn.Module) -> bool:
        self.val_epoch_losses.append(val_loss)
        self.train_epoch_losses.append(loss)
        wandb.log({"loss": loss, "val_loss": val_loss})

        return True

    def on_train_finish(self, model: torch.nn.Module):
        model.load_state_dict(self.best_state_dict)
