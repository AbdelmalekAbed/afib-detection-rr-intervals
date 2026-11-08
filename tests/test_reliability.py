"""Regression tests for entropy, data provenance and independent outer predictions."""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from torch import nn

from src.data.artifacts import load_windows, save_windows, window_filename
from src.data.loader import download_dataset
from src.features.hrv import shannon_entropy
from src.utils.artifacts import require_new_directory
from src.utils.splits import patient_kfold
from src.validation import choose_threshold, fit_torch, nested_torch_oof


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def dataset():
    rng = np.random.default_rng(4)
    X = rng.uniform(0.4, 1.4, size=(48, 8)).astype(np.float32)
    y = np.tile([0, 1, 0, 1, 0, 1, 0, 1], 6).astype(np.int8)
    groups = np.repeat([f"p{k}" for k in range(6)], 8)
    return X, y, groups


def factory():
    return nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1), nn.Flatten(0))


def test_entropy_uses_probability_masses():
    assert shannon_entropy(np.full(60, .8)) == 0
    assert shannon_entropy(np.array([0., 0., 1., 1.]), bins=2) == pytest.approx(np.log(2))
    assert shannon_entropy(np.arange(16.), bins=16) == pytest.approx(np.log(16))
    assert shannon_entropy(np.array([])) == 0


def test_entropy_is_invariant_to_rr_units():
    rr = np.array([.4, .5, .5, .7, .8, .9])
    assert shannon_entropy(rr) == pytest.approx(shannon_entropy(rr * 1000))


def test_versioned_cache_refuses_mismatch_and_overwrite(tmp_path):
    X, y, groups = dataset()
    path = tmp_path / window_filename("test", 8, 4, "majority")
    metadata = {"window": 8, "stride": 4, "label_strategy": "majority", "min_rr": .3}
    save_windows(path, X, y, groups, metadata)
    read = load_windows(path, expected_window=8, expected_metadata={"stride": 4})
    np.testing.assert_array_equal(read["X"], X)
    assert read["metadata"]["schema_version"] == 1
    with pytest.raises(ValueError, match="forme"):
        load_windows(path, expected_window=60)
    with pytest.raises(ValueError, match="stride"):
        load_windows(path, expected_window=8, expected_metadata={"stride": 3})
    with pytest.raises(FileExistsError):
        save_windows(path, X, y, groups, metadata)


def test_legacy_cache_requires_explicit_opt_in(tmp_path):
    X, y, groups = dataset()
    path = tmp_path / "old.npz"
    np.savez(path, X=X, y=y, patient_id=groups)
    with pytest.raises(ValueError, match="historique"):
        load_windows(path, expected_window=8)
    with pytest.warns(UserWarning, match="historique"):
        assert load_windows(path, expected_window=8, allow_legacy=True)["metadata"]["legacy"]
    with pytest.raises(ValueError):
        load_windows(path, expected_window=8, allow_legacy=True, expected_metadata={"stride": 4})


def test_existing_run_directory_is_preserved(tmp_path):
    marker = tmp_path / "metrics.json"
    marker.write_text('{"historical": true}')
    with pytest.raises(FileExistsError):
        require_new_directory(tmp_path)
    assert json.loads(marker.read_text()) == {"historical": True}


def test_partial_download_is_not_mistaken_for_complete(tmp_path, monkeypatch):
    directory = tmp_path / "afdb"
    directory.mkdir()
    marker = directory / "04015.hea"
    marker.write_text("partial download")
    monkeypatch.setattr("wfdb.dl_database", lambda *a, **k: pytest.fail("Unexpected network request"))
    with pytest.raises(RuntimeError, match="incomplet"):
        download_dataset("afdb", tmp_path)
    assert marker.read_text() == "partial download"


def test_descriptive_threshold_study_weights_patients_not_folds(monkeypatch):
    from scripts import phase6_analysis as analysis
    monkeypatch.setattr(analysis, "THRESHOLD_GRID", np.array([.5]))
    y = np.tile([0, 1], 3)
    scores = np.array([.1, .9, .1, .9, .9, .1])
    groups = np.repeat(["a", "b", "c"], 2)
    fold = np.array([0, 0, 0, 0, 1, 1])
    result = analysis.threshold_study(y, scores, groups, fold)
    assert result["cross_fold_exploratory_f1"] == pytest.approx(2 / 3)
    assert "nested_f1" not in result
    assert result["calibration_protocol"] == "exploratory_cross_fold_NOT_nested"


def test_initialization_and_predictions_are_seeded_before_factory():
    X, y, _ = dataset()
    kwargs = dict(epochs=2, batch_size=8, lr=.01, weight_decay=.001, seed=7)
    first = fit_torch(factory, X, y, **kwargs)
    torch.rand(1000)
    second = fit_torch(factory, X, y, **kwargs)
    for name in first.state:
        torch.testing.assert_close(first.state[name], second.state[name], rtol=0, atol=0)


def test_learning_curves_seed_the_model_factory():
    from src.baselines.cv import train_with_history
    X, y, _ = dataset()
    first, history = train_with_history(factory, X[:32], y[:32], X[32:], y[32:],
                                         epochs=2, batch_size=8, seed=7, weight_decay=.001)
    torch.rand(100)
    second, repeated = train_with_history(factory, X[:32], y[:32], X[32:], y[32:],
                                           epochs=2, batch_size=8, seed=7, weight_decay=.001)
    np.testing.assert_array_equal(first, second)
    assert history.train_loss == repeated.train_loss


def test_outer_test_labels_cannot_change_its_model_epoch_or_threshold():
    X, y, groups = dataset()
    kwargs = dict(n_splits=3, inner_splits=2, epochs=2, batch_size=8, seed=7)
    first = nested_torch_oof(factory, X, y, groups, **kwargs)
    test_idx = next(patient_kfold(groups, y, n_splits=3))[1]
    changed_y = y.copy()
    changed_y[test_idx] = 1 - changed_y[test_idx]
    second = nested_torch_oof(factory, X, changed_y, groups, **kwargs)
    np.testing.assert_array_equal(first.y_score[test_idx], second.y_score[test_idx])
    np.testing.assert_array_equal(first.thresholds[test_idx], second.thresholds[test_idx])
    assert first.details[0] == second.details[0]
    for key in first.states[0]:
        torch.testing.assert_close(first.states[0][key], second.states[0][key], rtol=0, atol=0)
    for detail in first.details:
        outer_test = set(detail["test_patients"])
        assert not outer_test.intersection(detail["selection_patients"])
        for inner in detail["inner_folds"]:
            assert not outer_test.intersection(inner["train_patients"] + inner["validation_patients"])
            assert not set(inner["train_patients"]).intersection(inner["validation_patients"])


def test_hyperparameter_search_is_repeated_inside_each_outer_fold():
    X, y, groups = dataset()
    result = nested_torch_oof(
        factory, X, y, groups, n_splits=3, inner_splits=2, epochs=1, batch_size=8,
        suggest_params=lambda trial: {"lr": trial.suggest_float("lr", .001, .01)},
        parameterized_factory=lambda params: factory(), n_trials=2)
    assert len(result.details) == 3
    assert all(len(d["trials"]) == 2 for d in result.details)
    assert all(.001 <= d["effective_lr"] <= .01 for d in result.details)


def test_threshold_prefers_nearest_half_in_tie():
    y = np.array([0, 1, 0, 1])
    scores = np.array([.01, .99, .01, .99])
    assert choose_threshold(y, scores, np.array(["a", "a", "b", "b"])) == pytest.approx(.5)


def test_onnx_export_keeps_opset_weights_and_dynamic_batch(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    from src.models.cnn_lstm import CNNLSTM
    from src.models.compression import dynamic_quantize, export_onnx
    from src.validation import predict

    torch.manual_seed(4)
    model = CNNLSTM(cnn_channels=(4, 8), lstm_hidden=4, head_hidden=4).eval()
    path = export_onnx(model, 60, tmp_path / "model.onnx")
    assert onnx.load(str(path)).opset_import[0].version == 17
    assert list(tmp_path.iterdir()) == [path]
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    X = np.random.default_rng(4).normal(size=(3, 60)).astype(np.float32)
    logits = session.run(None, {"rr_window": X})[0]
    np.testing.assert_allclose(predict(model, X), 1 / (1 + np.exp(-logits)), atol=1e-6)
    assert np.isfinite(predict(dynamic_quantize(model), X)).all()


@pytest.mark.parametrize("kwargs", [{"epochs": 0}, {"batch_size": 0}, {"weight_decay": -.1}])
def test_invalid_training_settings_fail_before_training(kwargs):
    X, y, _ = dataset()
    params = dict(epochs=1, batch_size=8, lr=.01, weight_decay=0., seed=7)
    params.update(kwargs)
    with pytest.raises(ValueError):
        fit_torch(factory, X, y, **params)
