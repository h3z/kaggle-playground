import pandas as pd
import numpy as np
import warnings
import gc
from IPython.display import HTML

warnings.filterwarnings("ignore")

from math import sin, cos, pi

from timeit import default_timer as timer
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup
from tqdm.auto import tqdm
import random

batch_size = 64
random_state = 42
random.seed(42)
torch.manual_seed(42)
import os
from torchinfo import summary

os.chdir(
    "/home/yanhuize/kaggle/Tabular-Playground-Series-Apr-2022/notebook/from_kaggle"
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


PATH_TRAIN = "../../dataset/csv/train.csv"
PATH_LABELS = "../../dataset/csv/train_labels.csv"
PATH_TEST = "../../dataset/csv/test.csv"
PATH_SUBMISSION = "../../dataset/csv/sample_submission.csv"

data = pd.read_csv(PATH_TRAIN)
data_labels = pd.read_csv(PATH_LABELS)
test_data = pd.read_csv(PATH_TEST)
submission = pd.read_csv(PATH_SUBMISSION)

scaler = StandardScaler()
data = data.drop(["sequence", "subject", "step"], axis=1)
data = scaler.fit_transform(data)


test_q = 0.85

train_size = int(test_q * len(data) - (test_q * len(data) % 60))
train_label_size = int(test_q * len(data_labels))

X_train, y_train = data[:train_size], data_labels[:train_label_size]
X_test, y_test = data[train_size:], data_labels[train_label_size:]


epochs = 200
seq_num = 60
# device = "cuda"
device = 0
lr = 1e-4


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seq_num):
        super().__init__()
        self.X = X
        self.y = y
        self.seq_num = seq_num

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx[0] // self.seq_num]


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, X, seq_num):
        super().__init__()
        self.X = X
        self.seq_num = seq_num

    def __len__(self):
        return len(self.X) // 60

    def __getitem__(self, idx):
        return self.X[idx]


def prepare_data(data, data_labels, seq_num, data_num, mode="train"):
    if data_labels is not None:
        data_labels = data_labels["state"].values

    sampler = np.array(
        [
            list(range(i * seq_num, (i + 1) * seq_num))
            for i in range(data_num // seq_num)
        ]
    )
    if mode == "train":
        dataset = TrainDataset(data, data_labels, seq_num)
    else:
        dataset = TestDataset(data, seq_num)

    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)
    return dataloader


class LSTM(nn.Module):
    def __init__(
        self, seq_num=60, input_dim=13, lstm_dim=512, num_layers=2, num_classes=1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, lstm_dim, 6, batch_first=True, bidirectional=True
        )

        # self.lstm = nn.LSTM(
        #     input_dim, lstm_dim, num_layers, batch_first=True, bidirectional=True
        # )

        # self.lstm1 = nn.LSTM(
        #     2 * lstm_dim, lstm_dim, num_layers, batch_first=True, bidirectional=True
        # )

        # self.lstm2 = nn.LSTM(
        #     2 * lstm_dim, lstm_dim, num_layers, batch_first=True, bidirectional=True
        # )

        self.logits = nn.Sequential(
            nn.ReLU(),
            nn.Linear(lstm_dim * seq_num * 2, num_classes),
        )

    def forward(self, x):
        features, _ = self.lstm(x)
        # features, _ = self.lstm1(features)
        # features, _ = self.lstm2(features)
        features = features.reshape(features.shape[0], -1)
        pred = self.logits(features)
        return pred


def train(
    epochs, model, optimizer, criterion, sheduler, train_iterator, valid_iterator
):
    for epoch in range(1, epochs + 1):
        training_loss = 0.0
        valid_loss = 0.0
        model.train()

        for batch_idx, batch in tqdm(
            enumerate(train_iterator), total=len(train_iterator)
        ):
            optimizer.zero_grad()
            batch[0] = batch[0].to(device)
            predict = model(batch[0].float()).squeeze(-1)
            loss = criterion(predict, batch[1].to(device).float())
            loss.backward()
            optimizer.step()
            # sheduler.step()
            training_loss += loss.data.item()
        training_loss /= len(train_iterator)
        print("lr", optimizer.param_groups[0]["lr"])

        model.eval()

        for batch_idx, batch in enumerate(valid_iterator):
            batch[0] = batch[0].to(device)
            predict = model(batch[0].float()).squeeze(-1)
            loss = criterion(predict, batch[1].to(device).float())
            valid_loss += loss.data.item()

        valid_loss /= len(valid_iterator)

        if epoch % 10 == 1:
            print(
                "Epoch: {}, Training Loss: {:.5f}, "
                "Validation Loss: {:.5f}".format(epoch, training_loss, valid_loss)
            )


def predict(
    model,
    loader,
):
    model.eval()

    preds = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred = model(data.float())
            preds.append(pred.detach().cpu().numpy())

    preds = np.concatenate(preds, 0)

    return preds


train_dataloader = prepare_data(X_train, y_train, 60, X_train.shape[0])
test_dataloader = prepare_data(X_test, y_test, 60, X_test.shape[0])


model = LSTM()
summary(model, (128, 60, 13))

model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_warmup_steps = int(0.1 * epochs * len(train_dataloader))
num_training_steps = int(epochs * len(train_dataloader))

# sheduler = get_linear_schedule_with_warmup(
#     optimizer, num_warmup_steps, num_training_steps
# )
sheduler = None

# criterion = nn.BCEWithLogitsLoss()
criterion = nn.MSELoss()

train(epochs, model, optimizer, criterion, sheduler, train_dataloader, test_dataloader)
# torch.save(model.state_dict(), 'model_92_2.pt')
# # model.load_state_dict(torch.load('model_92.pt'))


# test
test_data = test_data.drop(["sequence", "subject", "step"], axis=1)
test_data = scaler.transform(test_data)
loader = prepare_data(test_data, None, 60, test_data.shape[0], "test")
pred = predict(model, loader)
submission["state"] = pd.DataFrame(pred)
submission.to_csv("submit.csv", index=False)
