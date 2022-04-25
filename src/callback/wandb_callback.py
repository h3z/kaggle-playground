import numpy as np
import torch
import wandb

from callback.callback import Callback
from sklearn import metrics


class WandbCallback(Callback):
    def __init__(self) -> None:
        self.train_epoch_losses = []
        self.val_epoch_losses = []
        self.train_batch_losses = []
        self.val_batch_losses = []

    def on_val_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        preds = preds.detach().cpu().numpy().squeeze()
        gts = gts.detach().cpu().numpy().squeeze()
        try:
            wandb.log({"val_auc": metrics.roc_auc_score(preds > 0.5, gts)})
        except:
            wandb.log({"val_auc": 0})
            return

    def on_train_batch_end(self, preds: np.ndarray, gts: np.ndarray, loss):
        self.train_batch_losses.append(loss)

    def on_epoch_end(self, loss, val_loss, model: torch.nn.Module) -> bool:
        self.val_epoch_losses.append(val_loss)
        self.train_epoch_losses.append(loss)
        wandb.log({"loss": loss, "val_loss": val_loss})

        return True
