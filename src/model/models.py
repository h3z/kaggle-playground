import torch
from model import lstm


def get(device="cuda") -> torch.nn.Module:
    return lstm.LSTM().to(device)
