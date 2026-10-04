import numpy as np
import pytest

from tests.utils import DummyContext, make_hit, make_records
from waveform_analysis.core.plugins.builtin.cpu.hit_merged_features import (
    HIT_MERGED_FEATURES_DTYPE,
    HitMergedFeaturesPlugin,
)
from waveform_analysis.core.plugins.builtin.cpu.peaklet_channels import (
    PEAKLET_CHANNELS_DTYPE,
    PeakletChannelsPlugin,
)
from waveform_analysis.core.plugins.builtin.cpu.peaklets import (
    PEAKLET_COMPONENTS_DTYPE,
    PEAKLET_DTYPE,
    PEAKLET_FEATURES_DTYPE,
)
from waveform_analysis.core.plugins.builtin.hit.hit_finder import THRESHOLD_HIT_DTYPE
from waveform_analysis.core.plugins.builtin.hit.hit_merge import (
    HIT_MERGED_COMPONENTS_DTYPE,
    HIT_MERGED_DTYPE,
)
from waveform_analysis.core.plugins.builtin.peaklet_features.plugin import PeakletFeaturesPlugin
from waveform_analysis.core.plugins.builtin.peaklet_waveforms.plugin import PeakletWaveformPlugin
from waveform_analysis.core.plugins.builtin.shared.waveform_merge import (
    WaveformOverlapConflictError,
)


def _peaklets(areas, *, component_count=1):
    out = np.zeros(len(areas), dtype=PEAKLET_DTYPE)
    out["component_count"] = component_count
    return out


def _peaklet_features(areas):
    out = np.zeros(len(areas), dtype=PEAKLET_FEATURES_DTYPE)
    out["peak_id"] = np.arange(len(areas), dtype=np.int64)
    out["area"] = np.asarray(areas, dtype=np.float32)
    return out


def _components(pairs):
    out = np.zeros(len(pairs), dtype=PEAKLET_COMPONENTS_DTYPE)
    for i, (peaklet_id, merged_index) in enumerate(pairs):
        out[i]["peak_id"] = peaklet_id
        out[i]["merged_index"] = merged_index
    return out


def _features(rows):
    out = np.zeros(len(rows), dtype=HIT_MERGED_FEATURES_DTYPE)
    for i, row in enumerate(rows):
        out[i]["merged_index"] = row["merged_index"]
        out[i]["board"] = row.get("board", 0)
        out[i]["channel"] = row["channel"]
        out[i]["area"] = row["area"]
        out[i]["height"] = row["height"]
        out[i]["n_hits"] = row["n_hits"]
        out[i]["valid"] = row.get("valid", 1)
    return out


def _ctx(peaklets, components, features, peaklet_features):
    return DummyContext(
        {},
        {
            "peaklets": peaklets,
            "peaklet_components": components,
            "hit_merged_features": features,
            "peaklet_features": peaklet_features,
        },
    )


def _compute(ctx):
    return PeakletChannelsPlugin()._compute_channels(
        peaklets=ctx._data["peaklets"],
        components=ctx._data["peaklet_components"],
        features=ctx._data["hit_merged_features"],
        peaklet_features=ctx._data["peaklet_features"],
    )


def test_peaklet_channels_single_peaklet_multiple_channels():
    ctx = _ctx(
        _peaklets([100.0], component_count=2),
        _components([(0, 0), (0, 1)]),
        _features(
            [
                {"merged_index": 0, "channel": 0, "area": 60.0, "height": 30.0, "n_hits": 2},
                {"merged_index": 1, "channel": 1, "area": 40.0, "height": 25.0, "n_hits": 1},
            ]
        ),
        _peaklet_features([100.0]),
    )

    out = _compute(ctx)

    assert out.dtype == PEAKLET_CHANNELS_DTYPE
    assert len(out) == 2
    np.testing.assert_array_equal(out["peaklet_id"], np.array([0, 0], dtype=np.int64))
    np.testing.assert_array_equal(out["channel"], np.array([0, 1], dtype=np.int16))
    np.testing.assert_allclose(out["area"], np.array([60.0, 40.0], dtype=np.float32))
    np.testing.assert_array_equal(out["n_hits"], np.array([2, 1], dtype=np.int32))
    np.testing.assert_allclose(out["area_fraction"], np.array([0.6, 0.4], dtype=np.float32))


def test_peaklet_channels_dense_feature_identity_matches_shuffled_feature_fallback():
    peaklets = _peaklets([100.0], component_count=2)
    components = _components([(0, 0), (0, 1)])
    peaklet_features = _peaklet_features([100.0])
    direct = _compute(
        _ctx(
            peaklets,
            components,
            _features(
                [
                    {"merged_index": 0, "channel": 0, "area": 60.0, "height": 30.0, "n_hits": 2},
                    {"merged_index": 1, "channel": 1, "area": 40.0, "height": 25.0, "n_hits": 1},
                ]
            ),
            peaklet_features,
        )
    )
    generic = _compute(
        _ctx(
            peaklets,
            components,
            _features(
                [
                    {"merged_index": 1, "channel": 1, "area": 40.0, "height": 25.0, "n_hits": 1},
                    {"merged_index": 0, "channel": 0, "area": 60.0, "height": 30.0, "n_hits": 2},
                ]
            ),
            peaklet_features,
        )
    )

    np.testing.assert_array_equal(direct, generic)


def test_peaklet_channels_presorted_components_do_not_need_lexsort(monkeypatch):
    import waveform_analysis.core.plugins.builtin.peaklet_channels.plugin as plugin_module

    ctx = _ctx(
        _peaklets([100.0], component_count=2),
        _components([(0, 0), (0, 1)]),
        _features(
            [
                {"merged_index": 0, "channel": 0, "area": 60.0, "height": 30.0, "n_hits": 2},
                {"merged_index": 1, "channel": 1, "area": 40.0, "height": 25.0, "n_hits": 1},
            ]
        ),
        _peaklet_features([100.0]),
    )

    monkeypatch.setattr(
        plugin_module.np,
        "lexsort",
        lambda *_args, **_kwargs: pytest.fail("presorted fast path called lexsort"),
    )
    out = _compute(ctx)

    assert len(out) == 2


def test_peaklet_channels_fast_csr_preserves_member_order_and_offsets():
    peaklets = _peaklets([100.0, 50.0], component_count=0)
    peaklets[0]["component_offset"] = 0
    peaklets[0]["component_count"] = 3
    peaklets[1]["component_offset"] = 3
    peaklets[1]["component_count"] = 2
    components = _components([(0, 0), (0, 1), (0, 2), (1, 3), (1, 4)])
    features = _features(
        [
            {"merged_index": 0, "channel": 1, "area": 10.0, "height": 2.0, "n_hits": 1},
            {"merged_index": 1, "channel": 1, "area": 20.0, "height": 5.0, "n_hits": 2},
            {"merged_index": 2, "channel": 3, "area": 30.0, "height": 4.0, "n_hits": 1},
            {"merged_index": 3, "channel": 0, "area": 15.0, "height": 3.0, "n_hits": 1},
            {"merged_index": 4, "channel": 2, "area": 35.0, "height": 7.0, "n_hits": 2},
        ]
    )
    out, group_offsets, grouped_members = PeakletChannelsPlugin()._compute_channels(
        peaklets=peaklets,
        components=components,
        features=features,
        peaklet_features=_peaklet_features([60.0, 50.0]),
        validate=False,
        return_groups=True,
    )

    np.testing.assert_array_equal(out["peaklet_id"], [0, 0, 1, 1])
    np.testing.assert_array_equal(out["channel"], [1, 3, 0, 2])
    np.testing.assert_allclose(out["area"], [30.0, 30.0, 15.0, 35.0])
    np.testing.assert_array_equal(group_offsets, [0, 2, 3, 4, 5])
    np.testing.assert_array_equal(grouped_members, [0, 1, 2, 3, 4])


def test_peaklet_channels_fast_csr_sorts_unsorted_components_per_peaklet():
    peaklets = _peaklets([100.0], component_count=0)
    peaklets[0]["component_offset"] = 0
    peaklets[0]["component_count"] = 4
    components = _components([(0, 2), (0, 0), (0, 3), (0, 1)])
    features = _features(
        [
            {"merged_index": 0, "channel": 2, "area": 20.0, "height": 2.0, "n_hits": 1},
            {"merged_index": 1, "channel": 0, "area": 10.0, "height": 5.0, "n_hits": 2},
            {"merged_index": 2, "channel": 1, "area": 30.0, "height": 4.0, "n_hits": 1},
            {"merged_index": 3, "channel": 0, "area": 40.0, "height": 7.0, "n_hits": 3},
        ]
    )
    out, group_offsets, grouped_members = PeakletChannelsPlugin()._compute_channels(
        peaklets=peaklets,
        components=components,
        features=features,
        peaklet_features=_peaklet_features([100.0]),
        validate=False,
        return_groups=True,
    )

    np.testing.assert_array_equal(out["channel"], [0, 1, 2])
    np.testing.assert_allclose(out["area"], [50.0, 30.0, 20.0])
    np.testing.assert_array_equal(out["n_hits"], [5, 1, 1])
    np.testing.assert_array_equal(group_offsets, [0, 2, 3, 4])
    np.testing.assert_array_equal(grouped_members, [3, 1, 2, 0])


def test_peaklet_channels_noncontiguous_components_use_generic_fallback(monkeypatch):
    import waveform_analysis.core.plugins.builtin.peaklet_channels.plugin as plugin_module

    peaklets = _peaklets([100.0, 50.0], component_count=0)
    peaklets[0]["component_offset"] = 0
    peaklets[0]["component_count"] = 1
    peaklets[1]["component_offset"] = 1
    peaklets[1]["component_count"] = 1
    components = _components([(1, 1), (0, 0)])
    features = _features(
        [
            {"merged_index": 0, "channel": 0, "area": 100.0, "height": 10.0, "n_hits": 1},
            {"merged_index": 1, "channel": 0, "area": 50.0, "height": 5.0, "n_hits": 1},
        ]
    )
    calls = {"count": 0}
    original_lexsort = plugin_module.np.lexsort

    def counting_lexsort(*args, **kwargs):
        calls["count"] += 1
        return original_lexsort(*args, **kwargs)

    monkeypatch.setattr(plugin_module.np, "lexsort", counting_lexsort)
    out = _compute(_ctx(peaklets, components, features, _peaklet_features([100.0, 50.0])))

    assert calls["count"] == 1
    np.testing.assert_array_equal(out["peaklet_id"], [0, 1])


def test_peaklet_channels_aggregates_multiple_rows_for_same_channel():
    ctx = _ctx(
        _peaklets([50.0], component_count=2),
        _components([(0, 0), (0, 1)]),
        _features(
            [
                {"merged_index": 0, "channel": 3, "area": 20.0, "height": 7.0, "n_hits": 1},
                {"merged_index": 1, "channel": 3, "area": 30.0, "height": 11.0, "n_hits": 2},
            ]
        ),
        _peaklet_features([50.0]),
    )

    out = _compute(ctx)

    assert len(out) == 1
    assert int(out[0]["channel"]) == 3
    assert float(out[0]["area"]) == 50.0
    assert float(out[0]["height"]) == 11.0
    assert int(out[0]["n_hits"]) == 3
    assert float(out[0]["area_fraction"]) == 1.0


def test_peaklet_channels_reconstructs_and_deduplicates_cross_record_waveform():
    peaklets = _peaklets([70.0])
    components = _components([(0, 0)])
    features = _features(
        [{"merged_index": 0, "channel": 3, "area": 100.0, "height": 30.0, "n_hits": 2}]
    )
    peaklet_features = _peaklet_features([70.0])
    merged = np.zeros(1, dtype=HIT_MERGED_DTYPE)
    merged[0]["board"] = 0
    merged[0]["channel"] = 3
    merged[0]["record_id"] = 0
    merged[0]["sample_start"] = -1
    merged[0]["sample_end"] = -1
    merged[0]["is_single_record"] = False
    hits = np.array(
        [
            make_hit(record_id=0, channel=3, edge_start=2, edge_end=4),
            make_hit(record_id=1, channel=3, edge_start=1, edge_end=3),
        ],
        dtype=THRESHOLD_HIT_DTYPE,
    )
    component_hits = np.array([(0, 0), (0, 1)], dtype=HIT_MERGED_COMPONENTS_DTYPE)
    records = make_records(n_records=2, event_length=10, baseline=100.0, dt=2)
    records["timestamp"] = [0, 4000]
    records["polarity"] = "negative"
    wave_pool = np.full(20, 100, dtype=np.uint16)
    wave_pool[[2, 3, 11, 12]] = [80, 80, 80, 70]

    ctx = DummyContext(
        {"wave_source": "records", "use_filtered": False, "clip_negative_signal": False},
        {
            "peaklets": peaklets,
            "peaklet_components": components,
            "hit_merged_features": features,
            "peaklet_features": peaklet_features,
            "hit_merged": merged,
            "hit_merged_components": component_hits,
            "hit_threshold": hits,
            "records": records,
            "wave_pool": wave_pool,
        },
    )
    out = PeakletChannelsPlugin().compute(ctx, "run_001")

    assert float(out[0]["area"]) == 70.0
    assert float(out[0]["height"]) == 30.0
    assert float(out[0]["area_fraction"]) == 1.0


def test_peaklet_channels_numba_conflict_reenters_python_canonical_oracle():
    peaklets = _peaklets([70.0])
    components = _components([(0, 0)])
    features = _features(
        [{"merged_index": 0, "channel": 3, "area": 100.0, "height": 30.0, "n_hits": 2}]
    )
    merged = np.zeros(1, dtype=HIT_MERGED_DTYPE)
    merged[0]["board"] = 0
    merged[0]["channel"] = 3
    merged[0]["record_id"] = 0
    merged[0]["sample_start"] = -1
    merged[0]["sample_end"] = -1
    merged[0]["is_single_record"] = False
    hits = np.array(
        [
            make_hit(record_id=0, channel=3, edge_start=2, edge_end=4),
            make_hit(record_id=1, channel=3, edge_start=1, edge_end=3),
        ],
        dtype=THRESHOLD_HIT_DTYPE,
    )
    component_hits = np.array([(0, 0), (0, 1)], dtype=HIT_MERGED_COMPONENTS_DTYPE)
    records = make_records(n_records=2, event_length=10, baseline=100.0, dt=2)
    records["timestamp"] = [0, 4000]
    records["polarity"] = "negative"
    wave_pool = np.full(20, 100, dtype=np.uint16)
    wave_pool[[2, 3, 11, 12]] = [80, 80, 60, 80]
    ctx = DummyContext(
        {"wave_source": "records", "use_filtered": False, "clip_negative_signal": False},
        {
            "peaklets": peaklets,
            "peaklet_components": components,
            "hit_merged_features": features,
            "peaklet_features": _peaklet_features([70.0]),
            "hit_merged": merged,
            "hit_merged_components": component_hits,
            "hit_threshold": hits,
            "records": records,
            "wave_pool": wave_pool,
        },
    )

    with pytest.raises(WaveformOverlapConflictError, match="conflicting overlap"):
        PeakletChannelsPlugin().compute(ctx, "run_001")


def test_peaklet_channels_reuses_single_record_feature_without_waveform_dependencies():
    peaklets = _peaklets([30.0])
    components = _components([(0, 0)])
    features = _features(
        [{"merged_index": 0, "channel": 3, "area": 30.0, "height": 20.0, "n_hits": 1}]
    )
    merged = np.zeros(1, dtype=HIT_MERGED_DTYPE)
    merged[0]["sample_start"] = 2
    merged[0]["sample_end"] = 4
    merged[0]["time_start"] = 4_000
    merged[0]["time_end"] = 8_000

    ctx = DummyContext(
        {},
        {
            "peaklets": peaklets,
            "peaklet_components": components,
            "hit_merged_features": features,
            "peaklet_features": _peaklet_features([30.0]),
            "hit_merged": merged,
        },
    )

    out = PeakletChannelsPlugin().compute(ctx, "run_001")

    assert len(out) == 1
    assert float(out[0]["area"]) == 30.0
    assert float(out[0]["height"]) == 20.0


@pytest.mark.parametrize("config_style", ["global", "nested", "namespaced"])
@pytest.mark.parametrize("normalize_to_pe", [False, True])
def test_peaklet_channels_keeps_adc_units_with_calibrated_hit_features(
    config_style, normalize_to_pe
):
    hit_config = {"normalize_to_pe": normalize_to_pe, "gain_adc_per_pe": {"0:0": 2.0}}
    if config_style == "nested":
        config = {"hit_merged_features": hit_config}
    elif config_style == "namespaced":
        config = {f"hit_merged_features.{key}": value for key, value in hit_config.items()}
    else:
        config = hit_config
    merged = np.zeros(1, dtype=HIT_MERGED_DTYPE)
    merged["sample_start"] = 1
    merged["sample_end"] = 4
    merged["time_start"] = 2_000
    merged["time_end"] = 8_000
    merged["dt"] = 2
    merged["component_count"] = 1
    merged["is_single_record"] = True
    records = make_records(n_records=1, event_length=5, baseline=100.0, dt=2)
    records["polarity"] = "negative"
    ctx = DummyContext(
        config,
        {
            "peaklets": _peaklets([60.0]),
            "peaklet_components": _components([(0, 0)]),
            "hit_merged": merged,
            "hit_merged_components": np.array([(0, 0)], dtype=HIT_MERGED_COMPONENTS_DTYPE),
            "hit_threshold": np.array(
                [make_hit(record_id=0, edge_start=1, edge_end=4)], dtype=THRESHOLD_HIT_DTYPE
            ),
            "records": records,
            "wave_pool": np.array([100, 90, 70, 80, 100], dtype=np.uint16),
        },
    )
    features = HitMergedFeaturesPlugin().compute(ctx, "run_001")
    ctx._data["hit_merged_features"] = features
    rows, pool = PeakletWaveformPlugin().build_pair(ctx, "run_001")
    ctx._data["peaklet_waveforms"] = rows
    ctx._data["peaklet_waveform_pool"] = pool
    ctx._data["peaklet_features"] = PeakletFeaturesPlugin().compute(ctx, "run_001")

    assert float(features[0]["area"]) == (30.0 if normalize_to_pe else 60.0)
    assert float(ctx._data["peaklet_features"][0]["area"]) == 60.0
    out = PeakletChannelsPlugin().compute(ctx, "run_001")

    assert float(out[0]["area"]) == 60.0
    assert float(out[0]["height"]) == 30.0
    assert float(out[0]["area_fraction"]) == 1.0


def test_peaklet_channels_rejects_invalid_features_that_break_area_conservation():
    ctx = _ctx(
        _peaklets([50.0, 100.0], component_count=2),
        _components([(1, 3), (0, 2), (1, 1), (0, 0)]),
        _features(
            [
                {
                    "merged_index": 3,
                    "board": 1,
                    "channel": 2,
                    "area": 60.0,
                    "height": 9.0,
                    "n_hits": 2,
                },
                {
                    "merged_index": 2,
                    "board": 0,
                    "channel": 4,
                    "area": 25.0,
                    "height": 4.0,
                    "n_hits": 1,
                },
                {
                    "merged_index": 1,
                    "board": 0,
                    "channel": 1,
                    "area": 40.0,
                    "height": 7.0,
                    "n_hits": 3,
                },
                {
                    "merged_index": 0,
                    "board": 0,
                    "channel": 3,
                    "area": 25.0,
                    "height": 5.0,
                    "n_hits": 1,
                    "valid": 0,
                },
            ]
        ),
        _peaklet_features([50.0, 100.0]),
    )

    with pytest.raises(ValueError, match="area conservation failed"):
        _compute(ctx)


def test_peaklet_channels_zero_peaklet_area_rejects_nonzero_channel_area():
    ctx = _ctx(
        _peaklets([0.0]),
        _components([(0, 0)]),
        _features(
            [
                {"merged_index": 0, "channel": 0, "area": 5.0, "height": 2.0, "n_hits": 1},
            ]
        ),
        _peaklet_features([0.0]),
    )

    with pytest.raises(ValueError, match="area conservation failed"):
        _compute(ctx)


def test_peaklet_channels_zero_peaklet_area_writes_zero_fractions_when_channels_cancel():
    ctx = _ctx(
        _peaklets([0.0], component_count=2),
        _components([(0, 0), (0, 1)]),
        _features(
            [
                {"merged_index": 0, "channel": 0, "area": 5.0, "height": 2.0, "n_hits": 1},
                {"merged_index": 1, "channel": 1, "area": -5.0, "height": 1.0, "n_hits": 1},
            ]
        ),
        _peaklet_features([0.0]),
    )

    out = _compute(ctx)

    np.testing.assert_array_equal(out["area_fraction"], np.zeros(2, dtype=np.float32))


def test_peaklet_channels_near_zero_nonzero_area_checks_fraction_sum():
    ctx = _ctx(
        _peaklets([1e-6]),
        _components([(0, 0)]),
        _features([{"merged_index": 0, "channel": 0, "area": 2e-6, "height": 1.0, "n_hits": 1}]),
        _peaklet_features([1e-6]),
    )

    with pytest.raises(ValueError, match="fraction conservation failed"):
        _compute(ctx)


def test_peaklet_channels_reject_components_misaligned_with_peaklets():
    peaklets = _peaklets([100.0])
    peaklets[0]["component_count"] = 2
    ctx = _ctx(
        peaklets,
        _components([(0, 0)]),
        _features(
            [
                {"merged_index": 0, "channel": 0, "area": 5.0, "height": 2.0, "n_hits": 1},
            ]
        ),
        _peaklet_features([100.0]),
    )

    with pytest.raises(ValueError, match="inconsistent with peaklets"):
        PeakletChannelsPlugin().compute(ctx, "run_001")
