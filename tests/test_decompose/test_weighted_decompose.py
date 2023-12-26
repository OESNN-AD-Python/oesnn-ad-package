import numpy as np

from oesnn_ad.decompose.weighted_decompose import WeightedCeemdan

STREAM = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test__merge():
    """
        Assert if weighted merge working properly.
    """
    ceemdan = WeightedCeemdan(STREAM, common_ratio=0.9)

    result = ceemdan._merge(np.array([
        np.array([False, False, True]),
        np.array([True, False, True]),
        np.array([True, False, False]),
        np.array([True, False, True]),
        np.array([False, False, True]),
        np.array([False, False, False]),
    ]))

    np.testing.assert_array_equal(result, np.array([True, False, True]))
