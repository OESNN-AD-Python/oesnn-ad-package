import numpy as np
import PyEMD
import pytest

from oesnn_ad.decompose.decompose import Decompose

STREAM = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test__merge():
    """
        Assert if merge threw an exception.
    """
    with pytest.raises(NotImplementedError):
        Decompose(STREAM)._merge(np.array([False, True, False]))


def test__decompose():
    """
        Assert if Decompose return proper value of spectrums.
    """
    decomp = Decompose(STREAM)
    PyEMD.CEEMDAN.noise_seed(decomp.ceemdan, 1000)
    assert decomp._decompose(2).shape[0] == 3
