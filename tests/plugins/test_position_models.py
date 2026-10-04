"""Seven-channel model routing, numerical anchors, and Context cache integration."""

import importlib

import numpy as np
import pytest

from tests.utils import DummyContext
from waveform_analysis import Context
from waveform_analysis.core.plugins.builtin.peaklet_channels import PEAKLET_CHANNELS_DTYPE
from waveform_analysis.core.plugins.builtin.position_reconstruction import (
    FLAG_EDGE_EVENT,
    FLAG_LOW_S2_SIGNAL,
    FLAG_POSITION_VALID,
    FLAG_XY_RECONSTRUCTED,
    FLAG_Z_RECONSTRUCTED,
    POSITION_RECONSTRUCTION_DTYPE,
)
from waveform_analysis.plugins import Plugin, PositionReconstructionPlugin

MODEL_CHANNELS = [[2, channel] for channel in [12, 11, 13, 14, 15, 9, 10]]
ANCHOR_COUNTS = [20, 50, 80, 110, 140, 170, 230]
PAIR_DTYPE = np.dtype(
    [
        ("pair_id", "i8"),
        ("s1_peak_id", "i8"),
        ("s2_peak_id", "i8"),
        ("selected", "?"),
        ("drift_time_ns", "f4"),
        ("s2_area", "f4"),
        ("s2_n_channels", "i2"),
    ]
)


def _pairs(peak_ids, *, areas=None):
    rows = np.zeros(len(peak_ids), dtype=PAIR_DTYPE)
    rows["pair_id"] = np.arange(len(rows)) + 100
    rows["s1_peak_id"] = 3
    rows["s2_peak_id"] = peak_ids
    rows["selected"] = True
    rows["drift_time_ns"] = 1000
    rows["s2_area"] = 800 if areas is None else areas
    rows["s2_n_channels"] = 7
    return rows


def _channels(entries):
    rows = np.zeros(len(entries), dtype=PEAKLET_CHANNELS_DTYPE)
    for index, (peak_id, board, channel, area) in enumerate(entries):
        rows[index]["peaklet_id"] = peak_id
        rows[index]["board"] = board
        rows[index]["channel"] = channel
        rows[index]["area"] = area
    return rows


def _config(method="Junshi", **overrides):
    return {
        "xy_method": method,
        "model_channels": MODEL_CHANNELS,
        "model_area_per_count": [1.0] * 7,
        **overrides,
    }


def _dummy(method, pairs, channels, **overrides):
    return DummyContext(
        _config(method, **overrides), {"s1_s2_pairs": pairs, "peaklet_channels": channels}
    )


class ModelPairs(Plugin):
    provides = "s1_s2_pairs"
    depends_on = []
    output_dtype = PAIR_DTYPE

    def compute(self, context, run_id, **kwargs):
        return _pairs([17])


class ModelChannels(Plugin):
    provides = "peaklet_channels"
    depends_on = []
    output_dtype = PEAKLET_CHANNELS_DTYPE

    def compute(self, context, run_id, **kwargs):
        return _channels(
            [
                (17, board, channel, area)
                for (board, channel), area in zip(MODEL_CHANNELS, ANCHOR_COUNTS, strict=True)
            ]
        )


def _context(storage_dir, method="Junshi", **overrides):
    config = {
        f"position_reconstruction.{key}": value
        for key, value in _config(method, **overrides).items()
    }
    context = Context(storage_dir=str(storage_dir), config=config)
    context.register(ModelPairs, ModelChannels, PositionReconstructionPlugin)
    return context


def _model_class(method):
    module = importlib.import_module(
        f"waveform_analysis.core.plugins.builtin.position_reconstruction.{method.lower()}"
    )
    return getattr(module, f"{method}Reconstructor")


@pytest.mark.parametrize(
    "variant,expected",
    [
        ("uniform", [-3.828542609557393, -20.832627287636885]),
        ("per_pmt", [-7.87731317626523, -17.322137493259245]),
    ],
)
def test_junshi_context_matches_original_joblib_predictions(tmp_path, variant, expected):
    # Original joblib.predict under sklearn 1.2.2, with integer ANCHOR_COUNTS.
    result = _context(tmp_path, junshi_variant=variant).get_data("run", "position_reconstruction")

    assert result["xy_method"].tolist() == ["Junshi"]
    np.testing.assert_allclose([result[0]["x"], result[0]["y"]], expected, atol=2e-6, rtol=0)
    assert result.dtype == POSITION_RECONSTRUCTION_DTYPE
    np.testing.assert_allclose(result["z"], [1.3])
    assert result["z_method"].tolist() == ["drift_time"]
    assert result[0]["flags"] & FLAG_POSITION_VALID


def test_jun_plugin_matches_archived_v3_simulation():
    counts = [792, 306, 386, 272, 317, 340, 327]
    channels = _channels(
        [
            (17, board, channel, area)
            for (board, channel), area in zip(MODEL_CHANNELS, counts, strict=True)
        ]
    )
    result = PositionReconstructionPlugin().compute(_dummy("Jun", _pairs([17]), channels), "run")

    assert result["xy_method"].tolist() == ["Jun"]
    np.testing.assert_allclose(
        [result[0]["x"], result[0]["y"]],
        [-2.0977899866483307, -1.1754429772637862],
        atol=2e-7,
        rtol=0,
    )


@pytest.mark.parametrize("method", ["Jun", "Junshi"])
def test_model_inputs_and_predictions_follow_selected_unique_s2_rows(monkeypatch, method):
    calls = []

    def predict(self, counts):
        calls.append(counts.copy())
        return counts[:, :2].copy()

    monkeypatch.setattr(_model_class(method), "predict", predict)
    pairs = _pairs([20, 10, 20, 30, 40, 50, 60, 70], areas=[800, 800, 800, 99, 800, 800, 800, 800])
    pairs["selected"][-1] = False
    pairs["drift_time_ns"] = 1000 * np.arange(1, 9)
    channels = _channels(
        [
            (20, 2, 12, 4),
            (20, 2, 12, 2),
            (20, 2, 11, 6),
            (20, 1, 12, 10000),
            (20, 2, 8, 10000),
            (10, 2, 12, 20),
            (10, 2, 13, -5),
            (30, 2, 12, 100),
            (40, 2, 12, -8),
            (50, 2, 12, np.nan),
            (50, 2, 11, 3),
            (70, 2, 12, 100),
        ]
    )
    ctx = _dummy(
        method, pairs, channels, model_area_per_count=[2, 3, 1, 1, 1, 1, 1], model_rotation_deg=90.0
    )
    result = PositionReconstructionPlugin().compute(ctx, "run")

    assert len(calls) == 1
    np.testing.assert_array_equal(
        sorted(calls[0].tolist()), [[3, 2, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0, 0]]
    )
    np.testing.assert_allclose(result["x"][:3], [-2, 0, -2], atol=1e-6)
    np.testing.assert_allclose(result["y"][:3], [3, 10, 3], atol=1e-6)
    assert result["xy_method"].tolist() == [method] * 3 + ["none"] * 4
    assert np.all(np.isnan(result["x"][3:]))
    assert np.all(np.isnan(result["y"][3:]))
    assert np.all(result["flags"][:3] & FLAG_XY_RECONSTRUCTED)
    assert np.all((result["flags"][3:] & FLAG_XY_RECONSTRUCTED) == 0)
    assert result[3]["flags"] & FLAG_LOW_S2_SIGNAL
    for field in ("x_err", "y_err", "xy_chi2", "position_goodness"):
        assert np.all(np.isnan(result[field]))
    assert np.all(result["xy_ndf"] == 0)
    assert result.dtype == POSITION_RECONSTRUCTION_DTYPE
    np.testing.assert_array_equal(result["pair_id"], np.arange(100, 107))
    np.testing.assert_allclose(result["z"], 1.3 * np.arange(1, 8), rtol=1e-6)
    assert np.all(result["flags"] & FLAG_Z_RECONSTRUCTED)


@pytest.mark.parametrize(
    "method,x,detector_radius,is_edge",
    [
        ("Jun", 38.0, 62.5, False),
        ("Junshi", 38.0, 62.5, True),
        ("Jun", 43.0, 62.5, True),
        ("Junshi", 43.0, 62.5, True),
        ("Jun", 29.0, 30.0, True),
        ("Junshi", 29.0, 30.0, True),
    ],
)
def test_model_and_detector_radii_set_edge_flag_without_clipping(
    monkeypatch, method, x, detector_radius, is_edge
):
    monkeypatch.setattr(
        _model_class(method), "predict", lambda self, counts: np.tile([x, 0.0], (len(counts), 1))
    )
    ctx = _dummy(
        method,
        _pairs([17]),
        _channels([(17, 2, 12, 800)]),
        detector_radius_mm=detector_radius,
        edge_threshold_mm=2.0,
    )
    result = PositionReconstructionPlugin().compute(ctx, "run")

    assert result[0]["x"] == x
    assert result[0]["r"] == x
    assert bool(result[0]["flags"] & FLAG_EDGE_EVENT) is is_edge


@pytest.mark.parametrize("method", ["Jun", "Junshi"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("model_channels", None),
        ("model_channels", MODEL_CHANNELS[:6]),
        ("model_area_per_count", None),
        ("model_area_per_count", [1.0] * 6),
        ("model_area_per_count", [0.0] + [1.0] * 6),
    ],
)
def test_models_require_explicit_seven_channel_mapping_and_positive_scales(method, key, value):
    ctx = _dummy(method, _pairs([17]), _channels([(17, 2, 12, 800)]), **{key: value})

    with pytest.raises(ValueError, match=key):
        PositionReconstructionPlugin().compute(ctx, "run")


@pytest.mark.parametrize(
    "method,key,value",
    [
        ("Junshi", "xy_method", "Jun"),
        ("Junshi", "model_channels", MODEL_CHANNELS[::-1]),
        ("Junshi", "model_area_per_count", [2.0] + [1.0] * 6),
        ("Junshi", "model_rotation_deg", 90.0),
        ("Jun", "jun_qe", [0.32] * 7),
        ("Junshi", "junshi_variant", "per_pmt"),
    ],
)
def test_model_configuration_invalidates_memory_and_disk_position_cache(
    tmp_path, method, key, value
):
    context = _context(tmp_path, method)
    first_lineage = context.get_lineage("position_reconstruction")
    first = context.get_data("run", "position_reconstruction").copy()
    context.set_config({f"position_reconstruction.{key}": value})
    second_lineage = context.get_lineage("position_reconstruction")

    assert first_lineage != second_lineage
    second = context.get_data("run", "position_reconstruction")
    assert not np.allclose([first[0]["x"], first[0]["y"]], [second[0]["x"], second[0]["y"]])
    expected_method = value if key == "xy_method" else method
    assert second["xy_method"].tolist() == [expected_method]

    reopened = _context(tmp_path, method, **{key: value})
    assert reopened.get_lineage("position_reconstruction") == second_lineage
    reloaded = reopened.get_data("run", "position_reconstruction")
    for field in ("x", "y", "z", "r", "flags", "xy_method"):
        np.testing.assert_array_equal(reloaded[field], second[field])
