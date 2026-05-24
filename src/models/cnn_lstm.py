"""Hybrid CNN-LSTM model for AFib detection from RR-interval windows."""
from __future__ import annotations

import torch
from torch import nn


class CNNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, pool: int, dropout: float, batch_norm: bool):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
        ]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_ch))
        layers += [nn.ReLU(inplace=True), nn.MaxPool1d(pool), nn.Dropout(dropout)]
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNLSTM(nn.Module):
    """CNN → BiLSTM → pooling → MLP head.

    Designed to stay under ~100k parameters with default config.
    """

    def __init__(
        self,
        input_length: int = 30,
        cnn_channels: tuple[int, ...] = (16, 32),
        kernel_size: int = 5,
        pool_size: int = 2,
        cnn_dropout: float = 0.2,
        batch_norm: bool = True,
        lstm_hidden: int = 32,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        lstm_dropout: float = 0.2,
        head_hidden: int = 32,
        head_dropout: float = 0.3,
        out_classes: int = 1,
    ):
        super().__init__()
        blocks: list[nn.Module] = []
        prev = 1
        for ch in cnn_channels:
            blocks.append(CNNBlock(prev, ch, kernel_size, pool_size, cnn_dropout, batch_norm))
            prev = ch
        self.cnn = nn.Sequential(*blocks)

        self.lstm = nn.LSTM(
            input_size=prev,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out = lstm_hidden * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(lstm_out, head_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, out_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)
        return self.head(x).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
