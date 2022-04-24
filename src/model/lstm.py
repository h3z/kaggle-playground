from torch import nn


class LSTM(nn.Module):
    def __init__(
        self, seq_num=60, input_dim=13, lstm_dim=512, num_layers=2, num_classes=1
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim, lstm_dim, 6, batch_first=True, bidirectional=True
        )

        self.logits = nn.Sequential(
            nn.ReLU(),
            nn.Linear(lstm_dim * seq_num * 2, num_classes),
        )

    def forward(self, x):
        features, _ = self.lstm(x)
        features = features.reshape(features.shape[0], -1)
        pred = self.logits(features)
        return pred
