import numpy as np
import pytest

from oesnn_ad.decompose.decompose import Decompose

STREAM = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])

def test__merge():
    """
        Assert if Decompose threw an exception.
    """
    with pytest.raises(NotImplementedError):
        Decompose(STREAM)._merge(np.array([False, True, False]))
