"""Model compression utilities — quantization, pruning, ONNX export."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import prune


def dynamic_quantize(model: nn.Module) -> nn.Module:
    """Apply post-training dynamic quantization to Linear and LSTM layers."""
    return torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.Linear, nn.LSTM},
        dtype=torch.qint8,
    )


def magnitude_prune(model: nn.Module, amount: float = 0.3) -> nn.Module:
    """Global unstructured magnitude pruning on weights of Linear and Conv1d layers."""
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            parameters_to_prune.append((module, "weight"))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    for module, name in parameters_to_prune:
        prune.remove(module, name)
    return model


def export_onnx(model: nn.Module, input_length: int, out_path: Path) -> Path:
    """Export the model to ONNX for cross-runtime deployment."""
    model.eval()
    dummy = torch.zeros(1, input_length)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["rr_window"],
        output_names=["logit"],
        dynamic_axes={"rr_window": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
    )
    return out_path


def file_size_kb(path: Path) -> float:
    return path.stat().st_size / 1024.0
