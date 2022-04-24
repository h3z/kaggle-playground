import torch
from transformers import get_linear_schedule_with_warmup
import wandb


def get(
    optimizer: torch.optim.Optimizer, train_dataloader: torch.utils.data.DataLoader
):
    epochs = wandb.config["~epochs"]
    num_training_steps = int(epochs * len(train_dataloader))

    return get_linear_schedule_with_warmup(
        optimizer, int(0.1 * num_training_steps), num_training_steps
    )
