from http.client import ImproperConnectionState
import torch
import random
import tempfile


def fix_random():
    random.seed(42)
    torch.manual_seed(42)


def mktemp(f):
    return f"{tempfile.mkdtemp()}/{f}"
