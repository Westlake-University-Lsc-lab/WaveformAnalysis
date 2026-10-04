"""Position results must follow geometry and gain configuration changes."""

import numpy as np
import pytest

from waveform_analysis import Context
from waveform_analysis.plugins import Plugin, PositionReconstructionPlugin


class SelectedPair(Plugin):
    provides = "s1_s2_pairs"
    depends_on = []
    output_dtype = np.dtype(
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

    def compute(self, context, run_id, **kwargs):
        return np.array([(0, 1, 2, True, 1000, 200, 2)], dtype=self.output_dtype)


class PairChannels(Plugin):
    provides = "peaklet_channels"
    depends_on = []
    output_dtype = np.dtype(
        [("peaklet_id", "i8"), ("board", "i2"), ("channel", "i2"), ("area", "f4")]
    )

    def compute(self, context, run_id, **kwargs):
        return np.array([(2, 0, 1, 100), (2, 0, 2, 100)], dtype=self.output_dtype)


def make_context(storage_dir, x=20.0, gain=1.0):
    context = Context(
        storage_dir=str(storage_dir),
        config={
            "detector_geometry": {
                "pmt_mapping": [
                    {
                        "board": 0,
                        "channel": 1,
                        "pmt_no": 1,
                        "pmt_id": "A",
                        "x_mm": 0.0,
                        "y_mm": 0.0,
                        "gain": 1.0,
                    },
                    {
                        "board": 0,
                        "channel": 2,
                        "pmt_no": 2,
                        "pmt_id": "B",
                        "x_mm": x,
                        "y_mm": 0.0,
                        "gain": gain,
                    },
                ]
            }
        },
    )
    context.register(SelectedPair, PairChannels, PositionReconstructionPlugin)
    return context


@pytest.mark.parametrize("changed,expected", [({"x": 60.0}, 30.0), ({"gain": 3.0}, 5.0)])
def test_geometry_change_does_not_reuse_disk_position(tmp_path, changed, expected):
    first = make_context(tmp_path)
    np.testing.assert_allclose(first.get_data("run", "position_reconstruction")["x"], [10.0])
    second = make_context(tmp_path, **changed)
    assert first.get_lineage("position_reconstruction") != second.get_lineage(
        "position_reconstruction"
    )
    np.testing.assert_allclose(second.get_data("run", "position_reconstruction")["x"], [expected])


def test_set_config_updates_layout_on_same_plugin_instance(tmp_path):
    context = make_context(tmp_path)
    plugin = context._plugins["position_reconstruction"]
    np.testing.assert_allclose(plugin.compute(context, "run")["x"], [10.0])
    changed = make_context(tmp_path / "other", x=60.0)
    context.set_config({"detector_geometry": changed.config["detector_geometry"]})
    np.testing.assert_allclose(plugin.compute(context, "run")["x"], [30.0])
