"""Reference predictions from Junshi's original sklearn 1.2.2 models."""

import numpy as np
import pytest

from waveform_analysis.core.plugins.builtin.position_reconstruction.junshi import (
    JunshiReconstructor,
)

COUNTS = np.array(
    [
        [800 / 7] * 7,
        [800, 0, 0, 0, 0, 0, 0],
        [20, 50, 80, 110, 140, 170, 230],
        [0.02, 0.05, 0.08, 0.11, 0.14, 0.17, 0.23],
    ]
)
EXPECTED = {
    "uniform": [
        [2.4379885815624722, 4.294983657949018],
        [-4.552763672539324, -0.03154205724102002],
        [-3.828542609557393, -20.832627287636885],
        [-5.731791026962879, -14.595036674251448],
    ],
    "per_pmt": [
        [-0.23259037738460514, 4.481185369497088],
        [-3.3929375814745377, -3.5212977986036393],
        [-7.87731317626523, -17.322137493259245],
        [-8.852480359758259, -9.566057614883135],
    ],
}


@pytest.mark.parametrize("variant", EXPECTED)
def test_original_model_predictions(variant):
    reco = JunshiReconstructor(variant)

    np.testing.assert_allclose(reco.predict(COUNTS), EXPECTED[variant], atol=1e-10, rtol=0)
    assert reco.radius_mm == 39.5


@pytest.mark.parametrize("variant", EXPECTED)
def test_empty_batch(variant):
    assert JunshiReconstructor(variant).predict(np.empty((0, 7))).shape == (0, 2)


@pytest.mark.parametrize("variant", EXPECTED)
def test_multiple_batches_keep_event_order(variant):
    counts = np.tile(COUNTS, (129, 1))[:515]
    expected = np.tile(EXPECTED[variant], (129, 1))[:515]

    np.testing.assert_allclose(
        JunshiReconstructor(variant).predict(counts), expected, atol=1e-10, rtol=0
    )
