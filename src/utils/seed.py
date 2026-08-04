"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """Fixe les graines aléatoires de Python, NumPy et PyTorch.

    ``deterministic=True`` impose en plus des noyaux déterministes à PyTorch.
    Ce n'est pas le défaut car certains opérateurs n'ont pas d'implémentation
    déterministe et lèveraient alors une exception ; les expériences du dépôt
    tournent sur CPU, où le résultat est reproductible à la graine près sans
    cette contrainte (vérifié : la phase 2 rejouée redonne les mêmes chiffres).

    Attention : la graine ne pilote **pas** le découpage en plis. Celui-ci est
    assuré par ``GroupKFold``, qui est déterministe et n'utilise aucun aléa
    (cf. ``src.utils.splits``).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic)
