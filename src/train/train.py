import torch
import numpy as np
from typing import List

from callback.callback import Callback

from tqdm.auto import tqdm


def epoch_train(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    train_loader,
    criterion,
    callbacks: List[Callback] = [],
):
    model.train()

    losses = []
    for i, batch in tqdm(
        enumerate(train_loader), total=len(train_loader), unit=" batch", desc="Epoch: "
    ):
        batch_x = batch[0].to(torch.float32).to("cuda")
        batch_y = batch[1].to(torch.float32).to("cuda")

        optimizer.zero_grad()
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())

        [cb.on_train_batch_end(pred_y, batch_y, loss.item()) for cb in callbacks]

    return np.mean(losses)


def epoch_val(
    model: torch.nn.Module, val_loader, criterion, callbacks: List[Callback] = []
):
    model.eval()

    losses = []
    for i, batch in enumerate(val_loader):
        batch_x = batch[0].to(torch.float32).to("cuda")
        batch_y = batch[1].to(torch.float32).to("cuda")
        pred_y = model(batch_x)
        loss = criterion(pred_y, batch_y)
        losses.append(loss.item())
        [cb.on_val_batch_end(pred_y, batch_y, loss.item()) for cb in callbacks]

    return np.mean(losses)


def predict(model: torch.nn.Module, test_loader):
    model.eval()
    preds = []
    gts = []
    for i, (batch_x, batch_y) in enumerate(test_loader):
        batch_x = batch_x.to(torch.float32).to("cuda")
        batch_y = batch_y.to(torch.float32).to("cuda")
        pred_y = model.forward(batch_x, batch_y)

        preds.append(pred_y.cpu().detach().numpy())
        gts.append(batch_y.cpu().detach().numpy())

    preds = np.array(preds)
    preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
    gts = np.array(gts)
    gts = gts.reshape(-1, gts.shape[-2], gts.shape[-1])
    return preds, gts
