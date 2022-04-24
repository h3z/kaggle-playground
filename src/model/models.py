import torch
from model import lstm


def get() -> torch.nn.Module:
    return lstm.LSTM().to("cuda")
