import torch
import random


def fix_random():
    random.seed(42)
    torch.manual_seed(42)
