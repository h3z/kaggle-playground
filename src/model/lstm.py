from torch import nn
import wandb


class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        lstm_dim = wandb.config.hidden_size
        bidirectional = wandb.config.bidirectional
        layers = wandb.config.layer
        seq_num = 60
        input_dim = 13

        self.lstm = nn.LSTM(
            input_dim,
            lstm_dim,
            layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.logits = nn.Sequential(
            nn.ReLU(),
            nn.Linear(lstm_dim * seq_num * 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        features, _ = self.lstm(x)
        features = features.reshape(features.shape[0], -1)
        pred = self.logits(features)
        return pred
