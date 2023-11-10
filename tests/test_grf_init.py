"""
    Module tests GRFinit's methods.
"""
# pylint: disable=W0212

import numpy as np

from oesnn_ad.grf_init import GRFInit

WINDOW = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test__get_width_vec():
    """
        Assert if GRF width is calculated correctly.
    """
    grf = GRFInit(WINDOW, 7, 1, 0.5, 1.6)

    result = grf._get_width_vec()
    correct = np.repeat(0.11, 7)
    np.testing.assert_array_almost_equal(result, correct, decimal=2)


def test__get_width_vec_with_zero_value():
    """
        Assert if GRF width is calculated correctly when value is zero.
    """
    grf = GRFInit(np.array([0.5, 0.5, 0.5]), 7, 1, 0.5, 1.6)

    result = grf._get_width_vec()
    correct = np.repeat(1.0, 7)
    np.testing.assert_array_almost_equal(result, correct, decimal=2)


def test__get_center_vec():
    """
        Assert if GRF center is calculated correctly.
    """
    grf = GRFInit(WINDOW, 7, 1, 0.5, 1.6)

    result = grf._get_center_vec()
    correct = np.array([-0.17,  0.01,  0.19,  0.37,  0.55,  0.73,  0.91])
    np.testing.assert_array_almost_equal(result, correct, decimal=2)


def test__get_excitation():
    """
        Assert if GRF excitation is calculated correctly.
    """
    grf = GRFInit(WINDOW, 7, 1, 0.5, 1.6)
    width_v = np.repeat(0.18, 7)
    center_v = np.array([-0.17,  0.01,  0.19,  0.37,  0.55,  0.73,  0.91])

    result = grf._get_excitation(width_v, center_v)
    correct = np.array([0.001, 0.024, 0.227, 0.770, 0.962, 0.442, 0.074])
    np.testing.assert_array_almost_equal(result, correct, decimal=3)


def test__get_firing_time():
    """
        Assert if GRF firing time is calculated correctly.
    """
    grf = GRFInit(WINDOW, 7, 1, 0.5, 1.6)
    excitation = np.array([0.001, 0.024, 0.227, 0.770, 0.962, 0.442, 0.074])

    result = grf._get_firing_time(excitation)
    correct = np.array([0.999, 0.976, 0.773, 0.230, 0.038, 0.558, 0.926])
    np.testing.assert_array_almost_equal(result, correct, decimal=3)


def test__get_order():
    """
        Assert if neurons firing order is calculated correctly.
    """
    grf = GRFInit(WINDOW, 7, 1, 0.5, 1.6)
    firing_time = np.array([0.999, 0.976, 0.773, 0.230, 0.038, 0.558, 0.926])

    result = grf._get_order(firing_time)
    correct = np.array([6, 5, 3, 1, 0, 2, 4])
    np.testing.assert_array_equal(result, correct)


def test_get_order():
    """
        Assert if flow of GRF is working correctly.
    """
    grf = GRFInit(WINDOW, 7, 1, 0.5, 1.6)

    result = grf.get_order()
    correct = np.array([6, 5, 3, 1, 0, 2, 4])
    np.testing.assert_array_equal(result, correct)
