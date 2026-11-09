"""Tests del pipeline de clasificación con datos sintéticos (no requieren el .mat)."""

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dualidr import classification
from dualidr.classification import ClassificationParams, run_classification_pipeline, split_windows
from dualidr.data import RawRecords, WindowMeta, extract_features, make_windows

FS = 5000.0
N_CHANNELS = 5


def _make_raw(seed: int = 0, n_samples: int = 6000) -> RawRecords:
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / FS
    signals = []
    for r in range(3):
        sig = rng.normal(size=(n_samples, N_CHANNELS))
        if r == 0:  # registro "anómalo": tono de 100 Hz en el canal 1
            sig[:, 1] += 3 * np.sin(2 * np.pi * 100 * t)
        signals.append(sig)
    return RawRecords(signals, ["b_rec", "a_rec", "c_rec"], [f"ch{i}" for i in range(N_CHANNELS)])


def _labels(raw: RawRecords, positive: str = "b_rec") -> Tuple[WindowMeta, np.ndarray]:
    meta = make_windows(raw)
    y = np.array([raw.record_names[r] == positive for r in meta.record_id], dtype=int)
    return meta, y


def test_windows_follow_global_grid_sorted_by_name() -> None:
    raw = _make_raw()
    meta = make_windows(raw)
    assert np.all(meta.start % 160 == 0)
    assert np.all(meta.end - meta.start == 400)
    assert np.array_equal(meta.window_id, np.arange(len(meta)))
    assert [raw.record_names[r] for r in dict.fromkeys(meta.record_id)] == ["a_rec", "b_rec", "c_rec"]


def test_extract_features_places_tone_in_expected_column() -> None:
    raw = _make_raw()
    meta = make_windows(raw)
    feats = extract_features(raw.signals, meta)
    rows = meta.record_id == 0
    mean_per_col = feats[rows].mean(axis=0)
    assert feats.shape == (len(meta), 20 * N_CHANNELS)
    assert np.argmax(mean_per_col) == (100 // 25 - 1) * N_CHANNELS + 1


def test_split_has_no_leakage() -> None:
    raw = _make_raw()
    meta, y = _labels(raw)
    params = ClassificationParams(min_per_class=2)
    result = run_classification_pipeline(raw, meta, y, params)
    assert result.ok
    train, test = split_windows(meta, result.cuts, params.gap_samples)
    assert not np.any(train & test)
    for record, cut in result.cuts.items():
        rows = meta.record_id == record
        assert np.all(meta.end[rows & train] <= cut)
        assert np.all(meta.start[rows & test] >= cut + params.gap_samples)
    assert result.n_train + result.n_test + result.n_excluded == len(meta)
    assert result.f1 > 0.9  # la anomalía es claramente separable
    assert result.coef is not None and len(result.coef) == 20 * N_CHANNELS


def test_single_class_is_rejected() -> None:
    raw = _make_raw()
    meta, y = _labels(raw)
    result = run_classification_pipeline(raw, meta, np.zeros_like(y), ClassificationParams())
    assert not result.ok and result.coef is None


def test_missing_class_in_train_is_rejected() -> None:
    raw = _make_raw()
    meta = make_windows(raw)
    y = np.zeros(len(meta), dtype=int)
    last_a = np.flatnonzero(meta.record_id == 0)[-3:]  # positivos solo al final de un registro
    y[last_a] = 1
    result = run_classification_pipeline(raw, meta, y, ClassificationParams(min_per_class=1))
    assert not result.ok and "lacks" in result.reason


def test_signal_scaler_only_sees_training_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _make_raw()
    meta, y = _labels(raw)
    seen = []
    original = classification.fit_signal_scaler

    def spy(segments):  # type: ignore[no-untyped-def]
        seen.extend(len(s) for s in segments)
        return original(segments)

    monkeypatch.setattr(classification, "fit_signal_scaler", spy)
    result = run_classification_pipeline(raw, meta, y, ClassificationParams(min_per_class=2))
    assert sorted(seen) == sorted(result.cuts.values())
    assert all(cut < raw.signals[r].shape[0] for r, cut in result.cuts.items())


def test_feature_idx_restricts_columns_and_zero_fills() -> None:
    raw = _make_raw()
    meta, y = _labels(raw)
    tone_col = (100 // 25 - 1) * N_CHANNELS + 1
    columns = np.array([tone_col, 0, 7])
    result = run_classification_pipeline(raw, meta, y, ClassificationParams(min_per_class=2), columns)
    assert result.ok and result.coef is not None
    assert len(result.coef) == 20 * N_CHANNELS
    assert np.count_nonzero(np.delete(result.coef, columns)) == 0
    assert np.argmax(np.abs(result.coef)) == tone_col
    assert result.f1 > 0.9


def test_empty_feature_idx_is_rejected() -> None:
    raw = _make_raw()
    meta, y = _labels(raw)
    result = run_classification_pipeline(raw, meta, y, ClassificationParams(), np.array([], dtype=int))
    assert not result.ok and result.reason == "no features selected"
