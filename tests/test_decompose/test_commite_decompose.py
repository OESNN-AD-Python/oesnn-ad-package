import numpy as np

from oesnn_ad.decompose.committee_decompose import CommiteeDecompose

STREAM = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test__merge():
    ceemdan = CommiteeDecompose(STREAM)

    result = ceemdan._merge(np.array([
        np.array([False, False, True]),
        np.array([True, False, True]),
        np.array([False, False, False]),
    ]))

    np.testing.assert_array_equal(result, np.array([False, False, True]))
