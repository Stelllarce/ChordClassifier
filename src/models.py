from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn


def get_model(name="rf", **kwargs):
    models = {
        "rf": RandomForestClassifier,
        "svm": SVC,
        "knn": KNeighborsClassifier,
        "lr": LogisticRegression,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")
    return models[name](**kwargs)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class ChordCNN(nn.Module):
    def __init__(self, n_mels=128, n_frames=97):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


class ChordRNN(nn.Module):
    def __init__(self, n_mels=128, hidden_dim=64, num_layers=2, dropout=0.3, rnn_type='lstm'):
        super().__init__()
        self.norm = nn.LayerNorm(n_mels)
        if rnn_type == 'lstm':
            self.rnn = nn.LSTM(n_mels, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers > 1 else 0,
                               bidirectional=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(n_mels, hidden_dim, num_layers=num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0,
                              bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = x.squeeze(1).permute(0, 2, 1)
        x = self.norm(x)
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return self.classifier(out)
