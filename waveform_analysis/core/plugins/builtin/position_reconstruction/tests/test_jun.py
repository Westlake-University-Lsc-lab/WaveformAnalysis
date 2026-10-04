"""Jun ML numerical checks using analytic patterns and upstream simulation."""

import numpy as np
import pytest

from waveform_analysis.core.plugins.builtin.position_reconstruction.jun import (
    JunReconstructor,
)


def _linear_lrf(xy):
    x, y = np.asarray(xy).T
    return (
        np.column_stack(
            [1 + x / 80, 1 - x / 80, 1 + y / 80, 1 - y / 80, x * 0 + 1, x * 0 + 1, x * 0 + 1]
        )
        / 7
    )


@pytest.fixture
def linear_lrf_path(tmp_path):
    x, y = np.meshgrid(np.arange(-40, 41, 2), np.arange(-40, 41, 2))
    xy = np.column_stack([x.ravel(), y.ravel()])
    path = tmp_path / "linear_lrf.csv"
    np.savetxt(
        path,
        np.column_stack([xy, _linear_lrf(xy)]),
        delimiter=",",
        header="x,y,f0,f1,f2,f3,f4,f5,f6",
        comments="",
    )
    return path


def test_recovers_subgrid_positions_with_explicit_qe(linear_lrf_path):
    xy = np.array([[12.3, -8.2], [-6.7, 15.4], [0.0, 0.0]])
    qe = np.array([0.25, 0.31, 0.29, 0.33, 0.28, 0.32, 0.35])
    counts = 10000 * _linear_lrf(xy) * qe
    model = JunReconstructor(linear_lrf_path, qe=qe)

    np.testing.assert_allclose(model.predict(counts), xy, atol=0.003, rtol=0)


def test_pattern_scale_and_chunk_boundary_preserve_positions(linear_lrf_path):
    xy = np.tile([12.3, -8.2], (257, 1))
    counts = _linear_lrf(xy) * np.arange(1, 258)[:, None]
    model = JunReconstructor(linear_lrf_path, qe=np.ones(7))

    np.testing.assert_allclose(model.predict(counts), xy, atol=0.003, rtol=0)


def test_excludes_positions_outside_source_radius(linear_lrf_path):
    model = JunReconstructor(linear_lrf_path, qe=np.ones(7))

    np.testing.assert_allclose(model.predict(_linear_lrf([[45.0, 0.0]])), [[40.0, 0.0]])
    assert model.radius_mm == 40.0


def test_bundled_model_reproduces_archived_v3_simulation():
    # First two Kr83m simulation events and archived v3 ML results from the source NAS.
    counts = np.array([[792, 306, 386, 272, 317, 340, 327], [316, 173, 581, 207, 697, 167, 196]])
    expected = [
        [-2.0977899866483307, -1.1754429772637862],
        [-28.285970767455595, 21.738841422818247],
    ]

    np.testing.assert_allclose(JunReconstructor().predict(counts), expected, atol=1e-9, rtol=0)


def test_empty_events_keep_xy_shape():
    assert JunReconstructor().predict(np.empty((0, 7))).shape == (0, 2)
