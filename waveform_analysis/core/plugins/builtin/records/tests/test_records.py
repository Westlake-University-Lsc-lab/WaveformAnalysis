import numpy as np
import pytest

from tests.utils import FakeContext
from waveform_analysis.core.plugins.builtin.records import RecordsPlugin
from waveform_analysis.core.plugins.builtin.records._compute import (
    get_records_bundle,
    get_records_bundle_cache_key,
)
from waveform_analysis.core.processing.dtypes import RECORDS_DTYPE
from waveform_analysis.core.processing.records_builder import RecordsBundle, RecordsBundleRef


@pytest.fixture
def raw_csv_context(tmp_path, monkeypatch):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    raw_file = tmp_path / "DataR_CH0@VX2730_demo.CSV"
    raw_file.write_text(
        "BOARD;CHANNEL;TIMETAG;ENERGY;ENERGYSHORT;FLAGS;PROBE_CODE;SAMPLES\n"
        "0;0;3000;0;0;0x4000;1;30;31\n"
        "0;0;1000;0;0;0x4000;1;10;11\n"
        "0;0;1000;0;0;0x4000;1;20;21\n",
        encoding="utf-8",
    )
    plugin = RecordsPlugin()
    ctx = FakeContext(
        config={
            "daq_adapter": "vx2730",
            "show_progress": False,
            "records": {
                "channel_workers": 1,
                "n_jobs": 1,
                "records_part_size": 1,
                "use_process_pool": False,
            },
        },
        plugins={"records": plugin},
    )
    ctx._set_data("run_001", "raw_files", [[str(raw_file)]])
    return ctx, raw_file


def test_records_dtype_and_empty_compute():
    plugin = RecordsPlugin()
    bundle = RecordsBundle(
        records=np.zeros(0, dtype=RECORDS_DTYPE),
        wave_pool=np.zeros(0, dtype=np.uint16),
    )
    ctx = FakeContext(config={"daq_adapter": "vx2730"}, plugins={"records": plugin})
    cache_key = get_records_bundle_cache_key(ctx, "run_001")
    ctx._set_data("run_001", cache_key, bundle)

    out = plugin.compute(ctx, "run_001")

    assert out.dtype == RECORDS_DTYPE
    assert len(out) == 0


def test_records_compute_returns_cached_bundle_records():
    plugin = RecordsPlugin()
    records = np.zeros(2, dtype=RECORDS_DTYPE)
    records["timestamp"] = np.array([1000, 2000], dtype=np.int64)
    bundle = RecordsBundle(records=records, wave_pool=np.zeros(0, dtype=np.uint16))
    ctx = FakeContext(config={"daq_adapter": "vx2730"}, plugins={"records": plugin})
    cache_key = get_records_bundle_cache_key(ctx, "run_001")
    ctx._set_data("run_001", cache_key, bundle)

    out = plugin.compute(ctx, "run_001")

    np.testing.assert_array_equal(out["timestamp"], records["timestamp"])


def test_records_depends_on_raw_files_for_vx2730():
    plugin = RecordsPlugin()
    ctx = FakeContext(config={"daq_adapter": "vx2730"}, plugins={"records": plugin})

    assert plugin.resolve_depends_on(ctx) == ["raw_files"]


def test_records_can_depend_on_st_waveforms_for_vx2730():
    plugin = RecordsPlugin()
    ctx = FakeContext(
        config={"daq_adapter": "vx2730", "records": {"input_source": "st_waveforms"}},
        plugins={"records": plugin},
    )

    assert plugin.resolve_depends_on(ctx) == ["st_waveforms"]


def test_records_rejects_st_waveforms_source_for_v1725():
    plugin = RecordsPlugin()
    ctx = FakeContext(
        config={"daq_adapter": "v1725", "records": {"input_source": "st_waveforms"}},
        plugins={"records": plugin},
    )

    with pytest.raises(ValueError, match="not supported for v1725"):
        plugin.resolve_depends_on(ctx)


def test_get_records_bundle_returns_cached_bundle():
    plugin = RecordsPlugin()
    fake_bundle = RecordsBundle(
        records=np.zeros(2, dtype=RECORDS_DTYPE),
        wave_pool=np.array([1, 2, 3, 4], dtype=np.uint16),
    )
    ctx = FakeContext(config={"daq_adapter": "vx2730"}, plugins={"records": plugin})
    cache_key = get_records_bundle_cache_key(ctx, "run_001")
    ctx._set_data("run_001", cache_key, fake_bundle)

    assert get_records_bundle(ctx, "run_001") is fake_bundle


@pytest.mark.parametrize("keep_on_disk", ["default", True, False, None])
def test_raw_files_bundle_residency_preserves_order_and_cleanup(raw_csv_context, keep_on_disk):
    ctx, raw_file = raw_csv_context
    if keep_on_disk != "default":
        ctx.config["records"]["keep_on_disk"] = keep_on_disk
    if keep_on_disk is not None:
        ctx.config["records"]["memory_budget_gb"] = 0.0

    bundle = get_records_bundle(ctx, "run_001")
    disk_backed = keep_on_disk == "default" or keep_on_disk is True
    assert isinstance(bundle, RecordsBundleRef if disk_backed else RecordsBundle)
    try:
        loaded = bundle.load_full() if disk_backed else bundle
        assert loaded.records.dtype == RECORDS_DTYPE
        np.testing.assert_array_equal(loaded.records["timestamp"], [1000, 1000, 3000])
        np.testing.assert_array_equal(loaded.records["record_id"], [0, 1, 2])
        for record, expected in zip(loaded.records, ([10, 11], [20, 21], [30, 31]), strict=True):
            start = int(record["wave_offset"])
            stop = start + int(record["event_length"])
            np.testing.assert_array_equal(loaded.wave_pool[start:stop], expected)
        if disk_backed:
            assert bundle.temp_dir.is_dir()
            assert set(raw_file.parent.iterdir()) == {raw_file, bundle.temp_dir}
    finally:
        if isinstance(bundle, RecordsBundleRef):
            bundle.cleanup()
    assert list(raw_file.parent.iterdir()) == [raw_file]


def test_raw_files_bundle_enforces_memory_budget_and_cleans_parts(raw_csv_context):
    ctx, raw_file = raw_csv_context
    ctx.config["records"].update(keep_on_disk=None, memory_budget_gb=0.0)

    with pytest.raises(MemoryError, match="memory_budget_gb"):
        get_records_bundle(ctx, "run_001")

    assert list(raw_file.parent.iterdir()) == [raw_file]


@pytest.mark.parametrize("keep_on_disk", [True, False, None])
def test_raw_files_empty_bundle_preserves_dtype_without_disk_files(
    tmp_path, monkeypatch, keep_on_disk
):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    ctx = FakeContext(
        config={"records": {"keep_on_disk": keep_on_disk, "memory_budget_gb": 0.0}},
        plugins={"records": RecordsPlugin()},
    )
    ctx._set_data("run_001", "raw_files", [[]])

    bundle = get_records_bundle(ctx, "run_001")

    assert isinstance(bundle, RecordsBundle)
    assert bundle.records.dtype == RECORDS_DTYPE
    assert len(bundle.records) == 0
    assert bundle.wave_pool.dtype == np.dtype(np.uint16)
    assert len(bundle.wave_pool) == 0
    assert list(tmp_path.iterdir()) == []
