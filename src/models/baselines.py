"""Baseline models — HRV-feature classifier and pure CNN / pure LSTM.

Each baseline must be evaluated under the same patient-level split as the main model.
"""
from __future__ import annotations

import torch
from sklearn.ensemble import RandomForestClassifier
from torch import nn


def build_hrv_baseline(n_estimators: int = 400, random_state: int = 42) -> RandomForestClassifier:
    """Random Forest on HRV features (see :mod:`src.features.hrv`)."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )


class CNNOnly(nn.Module):
    """Ablation baseline: CNN trunk + pooling head, no recurrent layer."""

    def __init__(self, input_length: int = 30, channels: tuple[int, ...] = (16, 32), kernel: int = 5):
        super().__init__()
        layers: list[nn.Module] = []
        prev = 1
        for ch in channels:
            layers += [
                nn.Conv1d(prev, ch, kernel, padding=kernel // 2),
                nn.BatchNorm1d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout(0.2),
            ]
            prev = ch
        self.cnn = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(prev, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.head(self.cnn(x)).squeeze(-1)


class LSTMOnly(nn.Module):
    """Ablation baseline: BiLSTM directly on the raw RR sequence."""

    def __init__(self, hidden: int = 32, layers: int = 1, bidirectional: bool = True):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0,
        )
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out, _ = self.lstm(x)
        return self.head(out.mean(dim=1)).squeeze(-1)
