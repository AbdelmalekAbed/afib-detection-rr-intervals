"""Optimization contracts: preprocessing, calibration and patient isolation."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from src.baselines.search import nested_hrv_oof
from src.utils.metrics import per_patient_f1
from src.utils.splits import patient_kfold
from src.validation import choose_threshold, nested_torch_oof, predict, preprocess_windows


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def data():
    rng = np.random.default_rng(12)
    return (rng.uniform(.5, 1.1, (48, 60)).astype(np.float32),
            np.tile([0, 1], 24).astype(np.int8), np.repeat([f"p{k}" for k in range(6)], 8))


def tiny(params=None):
    width = (params or {}).get("width", 2)
    return nn.Sequential(nn.Linear(60, width), nn.ReLU(), nn.Linear(width, 1), nn.Flatten(0))


def test_relative_rr_preserves_variability_and_is_unit_invariant():
    regular = np.tile([.78, .8, .82], 20)
    irregular = .8 + 4 * (regular - .8)
    X = np.stack([regular, irregular]).astype(np.float32)
    relative = preprocess_windows(X, "relative_per_window")
    assert relative[1].std() / relative[0].std() == pytest.approx(4, abs=1e-5)
    np.testing.assert_allclose(relative, preprocess_windows(1000 * X, "relative_per_window"), atol=1e-6)
    np.testing.assert_allclose(preprocess_windows(X)[0], preprocess_windows(X)[1], atol=1e-4)
    np.testing.assert_array_equal(preprocess_windows(np.ones((2, 60)), "relative_per_window"), 0)
    with pytest.raises(ValueError):
        preprocess_windows(X, "unknown")


def test_vectorized_threshold_matches_patient_metric_with_single_class_subjects():
    rng = np.random.default_rng(11)
    y = np.r_[rng.integers(0, 2, 200), np.ones(100)]
    groups = np.repeat(["a", "b", "c"], 100)
    scores = rng.uniform(0, 1, len(y))
    grid = np.linspace(.05, .95, 91)
    values = np.array([per_patient_f1(y, scores, groups, t)["mean_patient_f1"] for t in grid])
    ties = np.flatnonzero(np.isclose(values, values.max(), atol=1e-12, rtol=0))
    expected = grid[ties[np.argmin(np.abs(grid[ties] - .5))]]
    assert choose_threshold(y, scores, groups, grid) == expected


def test_optimized_outer_predictions_remain_independent_and_reloadable():
    X, y, groups = data()

    def suggest(trial):
        return {"normalization": trial.suggest_categorical("normalization", ["zscore_per_window", "relative_per_window"]),
                "use_pos_weight": trial.suggest_categorical("use_pos_weight", [False, True])}

    kwargs = dict(n_splits=3, inner_splits=2, epochs=1, batch_size=8, n_trials=2,
                  suggest_params=suggest, parameterized_factory=tiny,
                  objective_metric="calibrated_patient_f1", startup_trials=1)
    first = nested_torch_oof(tiny, X, y, groups, **kwargs)
    te = next(patient_kfold(groups, y, n_splits=3))[1]
    changed = y.copy()
    changed[te] = 1 - changed[te]
    second = nested_torch_oof(tiny, X, changed, groups, **kwargs)
    np.testing.assert_array_equal(first.y_score[te], second.y_score[te])
    np.testing.assert_array_equal(first.thresholds[te], second.thresholds[te])
    assert first.details[0] == second.details[0]
    model = tiny(first.details[0]["params"])
    model.load_state_dict(first.states[0])
    np.testing.assert_allclose(predict(model, preprocess_windows(X[te], first.details[0]["normalization"])),
                               first.y_score[te], atol=1e-7)


def test_oversized_candidates_are_pruned_before_training():
    X, y, groups = data()
    result = nested_torch_oof(
        tiny, X, y, groups, n_splits=3, inner_splits=2, epochs=1, batch_size=8,
        suggest_params=lambda t: {"width": t.suggest_categorical("width", [1, 100])},
        parameterized_factory=tiny, n_trials=2, parameter_budget=100,
        initial_params=[{"width": 100}, {"width": 1}])
    for detail in result.details:
        assert [t["state"] for t in detail["trials"]] == ["PRUNED", "COMPLETE"]
        assert detail["params"] == {"width": 1}


def test_hrv_hyperparameters_and_threshold_do_not_use_outer_targets():
    from sklearn.linear_model import LogisticRegression
    X, y, groups = data()
    kwargs = dict(n_splits=3, inner_splits=2, n_trials=2,
                  suggest=lambda t: {"C": t.suggest_float("C", .1, 1)},
                  model_factory=lambda params, seed: LogisticRegression(**params, random_state=seed))
    first = nested_hrv_oof(X[:, :4], y, groups, **kwargs)
    te = next(patient_kfold(groups, y, n_splits=3))[1]
    changed = y.copy()
    changed[te] = 1 - changed[te]
    second = nested_hrv_oof(X[:, :4], changed, groups, **kwargs)
    np.testing.assert_array_equal(first["scores"][te], second["scores"][te])
    np.testing.assert_array_equal(first["thresholds"][te], second["thresholds"][te])
    assert first["details"][0] == second["details"][0]


def test_training_cli_records_effective_budget_and_checkpoints(tmp_path, monkeypatch):
    from src import train
    monkeypatch.setattr(train, "build_windows", lambda *a, **kw: data())
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["train", "--config", "configs/train.yaml", "--out-dir", str(out),
                                     "--epochs", "1", "--n-splits", "3", "--inner-splits", "2", "--trials", "0"])
    train.main()
    metadata = json.loads((out / "metrics.json").read_text())
    assert metadata["config"]["training"]["epochs"] == 1
    assert metadata["config"]["protocol"]["trials"] == 0
    assert metadata["config"]["data"]["split"]["n_folds"] == 3
    assert len(list(out.glob("fold*.pt"))) == 3
    assert (out / "source_snapshot" / Path("src/train.py")).is_file()


def test_comparison_refuses_misaligned_patients_and_weights_them_equally():
    from scripts.summarize_optimization import aligned, compare
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.repeat(["a", "b", "c"], 2)
    d = {"y": y, "groups": groups, "fold": np.repeat([0, 1, 2], 2)}
    score = np.array([.1, .9, .1, .9, .9, .1])
    result, _ = compare({"model": (d, score, score >= .5)}, [])
    assert result["models"]["model"]["mean"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="groups"):
        aligned(d, {**d, "groups": groups[::-1]})


def test_optimized_compression_freezes_thresholds_and_original_checkpoints(tmp_path, monkeypatch):
    from scripts import compress_optimized as compression
    from src.data.artifacts import save_windows
    from src.models.search import model_from_params
    from src.train import load_config
    from src.utils.artifacts import file_digest

    X, y, groups = data()
    windows = tmp_path / "windows.npz"
    save_windows(windows, X, y, groups, {"dataset": "afdb", "window": 60, "stride": 30})
    run = tmp_path / "run"
    run.mkdir()
    params = {"n_cnn_blocks": 2, "cnn_first": 8, "kernel_size": 3, "cnn_dropout": .1,
              "lstm_hidden": 16, "lstm_layers": 1, "head_hidden": 16, "head_dropout": .2}
    scores = np.zeros(len(y), np.float32)
    folds = np.zeros(len(y), np.int8)
    thresholds = np.full(len(y), .6, np.float32)
    details, originals = [], {}
    for k, (_, te) in enumerate(patient_kfold(groups, y, n_splits=3)):
        mode = "relative_per_window" if k == 0 else "zscore_per_window"
        model = model_from_params(params, 60)
        path = run / f"fold{k}.pt"
        torch.save(model.state_dict(), path)
        originals[path] = file_digest(path)
        scores[te] = predict(model, preprocess_windows(X[te], mode))
        folds[te] = k
        details.append({"fold": k, "params": params, "normalization": mode, "threshold": .6})
    np.savez(run / "oof_scores.npz", y=y, groups=groups, fold=folds, scores=scores, thresholds=thresholds)
    (run / "metrics.json").write_text(json.dumps({"protocol": "nested_patient_cv_v1", "dataset": "afdb",
        "window": 60, "folds": details, "config": load_config(Path("configs/train.yaml")),
        "provenance": {"windows_sha256": file_digest(windows)}}))
    out = tmp_path / "compressed"
    monkeypatch.setattr(sys, "argv", ["compress", "--run", str(run), "--windows", str(windows), "--out-dir", str(out)])
    monkeypatch.setattr(compression, "latency", lambda *a, **kw: {"p50_ms": 0., "p95_ms": 0.})
    compression.main()
    with np.load(out / "scores.npz", allow_pickle=False) as result:
        np.testing.assert_array_equal(result["thresholds"], thresholds)
        np.testing.assert_allclose(result["fp32"], scores, atol=1e-7)
    assert all(file_digest(path) == digest for path, digest in originals.items())
    assert len(list(out.glob("*.onnx"))) == 3

    # The external loader must restore the selected preprocessing for every backend.
    from src import evaluate
    external_windows = tmp_path / "external.npz"
    save_windows(external_windows, X, y, groups, {"dataset": "ltafdb", "window": 60, "stride": 30,
                                                "label_strategy": "majority", "min_rr": .3, "max_rr": 2.})
    monkeypatch.setattr(evaluate, "plot_curves", lambda *args: None)
    predictions = {}
    for variant in ("fp32", "int8_dynamic", "onnx_fp32"):
        monkeypatch.setattr(sys, "argv", ["evaluate", "--config", "configs/train.yaml", "--in-dir", str(run),
            "--external", "ltafdb", "--windows", str(external_windows), "--variant", variant, "--compression-run", str(out)])
        evaluate.main()
        folder = next(run.glob(f"evaluation_{variant}_*"))
        with np.load(folder / "scores.npz", allow_pickle=False) as evaluated:
            predictions[variant] = evaluated["scores"]
        assert json.loads((folder / "eval_ltafdb.json").read_text())["variant"] == variant
    np.testing.assert_allclose(predictions["onnx_fp32"], predictions["fp32"], atol=1e-5)
    np.testing.assert_allclose(predictions["int8_dynamic"], predictions["fp32"], atol=.02)
