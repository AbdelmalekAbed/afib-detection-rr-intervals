"""Patient-level cross-validation runners for Phase 2 baselines.

Every baseline (rule, HRV+RF, CNN-only, LSTM-only) must be scored under the
*same* patient-grouped folds so that the comparison table is honest. The two
public helpers here both return out-of-fold (OOF) score vectors aligned with
the input ``y`` array; metric aggregation is left to the caller.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.splits import patient_kfold


@dataclass
class OOFResult:
    """Aligned with ``y``: one OOF score per sample, plus the fold each came from."""
    y_score: np.ndarray
    fold: np.ndarray


@dataclass
class TrainingHistory:
    """Per-epoch metrics for a single (train, val) split — used for learning curves."""
    epoch: list[int] = field(default_factory=list)
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_auroc: list[float] = field(default_factory=list)
    val_f1: list[float] = field(default_factory=list)


def zscore_per_window(X: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-row z-score — matches the preprocessing key in ``configs/data.yaml``."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + eps
    return ((X - mean) / std).astype(np.float32)


def crossval_sklearn_oof(
    model_factory: Callable,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
) -> OOFResult:
    """OOF probabilities from any sklearn classifier exposing ``predict_proba``."""
    y_score = np.full(len(y), np.nan, dtype=np.float32)
    fold = np.full(len(y), -1, dtype=np.int8)
    for k, (tr, va) in enumerate(patient_kfold(groups, y, n_splits=n_splits)):
        model = model_factory()
        model.fit(X[tr], y[tr])
        y_score[va] = model.predict_proba(X[va])[:, 1].astype(np.float32)
        fold[va] = k
    return OOFResult(y_score=y_score, fold=fold)


def _train_one_torch_fold(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    pos_weight: float | None,
    seed: int,
    weight_decay: float = 0.0,
    early_stop_metric: str = "loss",
) -> np.ndarray:
    """Train a torch model with BCE-with-logits + early stopping.

    ``early_stop_metric`` selects what to monitor on the validation fold:
    ``"loss"`` (default, lower-is-better) or ``"auroc"`` (higher-is-better).
    AUROC is smoother than loss when ``pos_weight`` is large, which matters
    when the loss curve is too noisy to early-stop reliably.
    """
    torch.manual_seed(seed)

    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr.astype(np.float32))
    Xv = torch.from_numpy(X_va).float()
    yv = torch.from_numpy(y_va.astype(np.float32))

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    pw = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight else None
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    if early_stop_metric not in {"loss", "auroc"}:
        raise ValueError(f"unknown early_stop_metric: {early_stop_metric}")
    best_metric = float("inf") if early_stop_metric == "loss" else -float("inf")
    best_scores: np.ndarray | None = None
    bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = float(crit(val_logits, yv).item())
            scores = torch.sigmoid(val_logits).cpu().numpy()

        if early_stop_metric == "loss":
            current = val_loss
            improved = current < best_metric - 1e-4
        else:
            try:
                current = float(roc_auc_score(y_va, scores))
            except ValueError:
                current = -float("inf")
            improved = current > best_metric + 1e-4

        if improved:
            best_metric = current
            best_scores = scores
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    return best_scores if best_scores is not None else scores


def train_with_history(
    model: nn.Module,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int = 20,
    batch_size: int = 512,
    lr: float = 1e-3,
    pos_weight: float | None = None,
    seed: int = 42,
    zscore: bool = True,
) -> tuple[np.ndarray, TrainingHistory]:
    """Train a single (train, val) split *without* early stopping, returning per-epoch history.

    Used to produce learning-curve figures for the final model.
    """
    torch.manual_seed(seed)
    if zscore:
        X_tr = zscore_per_window(X_tr)
        X_va = zscore_per_window(X_va)

    Xt = torch.from_numpy(X_tr).float()
    yt = torch.from_numpy(y_tr.astype(np.float32))
    Xv = torch.from_numpy(X_va).float()
    yv = torch.from_numpy(y_va.astype(np.float32))
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    pw = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight else None
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)

    hist = TrainingHistory()
    last_val_scores = np.zeros(len(y_va), dtype=np.float32)
    for ep in range(epochs):
        model.train()
        running, n = 0.0, 0
        for xb, yb in loader:
            optim.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()
            running += float(loss.item()) * len(xb)
            n += len(xb)
        train_loss = running / max(n, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(Xv)
            val_loss = float(crit(val_logits, yv).item())
            scores = torch.sigmoid(val_logits).cpu().numpy()
        last_val_scores = scores
        try:
            val_auc = float(roc_auc_score(y_va, scores))
        except ValueError:
            val_auc = float("nan")
        val_f1 = float(f1_score(y_va, (scores >= 0.5).astype(int), zero_division=0))

        hist.epoch.append(ep + 1)
        hist.train_loss.append(train_loss)
        hist.val_loss.append(val_loss)
        hist.val_auroc.append(val_auc)
        hist.val_f1.append(val_f1)
    return last_val_scores, hist


def crossval_torch_oof(
    model_factory: Callable[[], nn.Module],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    epochs: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 3,
    use_pos_weight: bool = True,
    seed: int = 42,
    weight_decay: float = 0.0,
    early_stop_metric: str = "loss",
) -> OOFResult:
    """OOF sigmoid scores from a torch baseline trained one fold at a time."""
    Xn = zscore_per_window(X)
    y_score = np.full(len(y), np.nan, dtype=np.float32)
    fold = np.full(len(y), -1, dtype=np.int8)
    for k, (tr, va) in enumerate(patient_kfold(groups, y, n_splits=n_splits)):
        n_pos = max(int(y[tr].sum()), 1)
        n_neg = max(int(len(y[tr]) - n_pos), 1)
        pw = n_neg / n_pos if use_pos_weight else None
        model = model_factory()
        scores = _train_one_torch_fold(
            model=model,
            X_tr=Xn[tr],
            y_tr=y[tr],
            X_va=Xn[va],
            y_va=y[va],
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            pos_weight=pw,
            seed=seed + k,
            weight_decay=weight_decay,
            early_stop_metric=early_stop_metric,
        )
        y_score[va] = scores
        fold[va] = k
    return OOFResult(y_score=y_score, fold=fold)
