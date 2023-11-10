"""
    Module test Neuron's class methods.
"""

import numpy as np
from pytest import approx

from oesnn_ad.neuron import OutputNeuron

def test_update_neuron():
    """
        Test assert if neuron is updated.
    """
    updated_neuron = OutputNeuron(
        np.array([0.25]*3), 0.25, 0.1, 1, 0.25, 0.75, 2)
    candidate_neuron = OutputNeuron(
        np.array([0.75]*3), 0.5, 0.3, 1, 0.75, 1.0, 2)

    updated_neuron.update_neuron(candidate_neuron)

    assert updated_neuron.modification_count == 2
    assert updated_neuron.output_value == approx(0.2)
    assert updated_neuron.addition_time == approx(0.5)
    np.testing.assert_array_almost_equal(
        updated_neuron.weights, np.array([0.5]*3), decimal=3)


def test__get_item__():
    """
        Test assert if neuron indexing is correctly returning values from weights
        vector.
    """
    neuron = OutputNeuron(
        np.array([0.25, 0.5, 0.75]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    assert neuron[0] == 0.25
    assert neuron[1] == 0.5
    assert neuron[2] == 0.75


def test_error_correction():
    """
        Test assert if error correction method correctly update output value.
    """
    output_value = 0.1
    ksi = 0.5
    window_head = 1.0

    neuron = OutputNeuron(
        np.array([0.25, 0.5, 0.75]), 0.25, output_value, 1, 0.25, 0.75, 2)

    neuron.error_correction(window_head, ksi)
    assert neuron.output_value == 0.55


def test_error_calc():
    """
        Test assert if error calculation working correctly.
    """
    output_value = 0.1
    window_head = 1.0

    neuron = OutputNeuron(
        np.array([0.25, 0.5, 0.75]), 0.25, output_value, 1, 0.25, 0.75, 2)

    assert neuron.error_calc(window_head) == 0.9


def test_update_psp():
    """
        Test assert if PSP is updated correctly.
    """
    psp = 0.5
    update_psp = 0.3

    neuron = OutputNeuron(
        np.array([0.25, 0.5, 0.75]), 0.25, 0.1, 1, 0.25, psp, 2)

    neuron.update_psp(update_psp)

    assert neuron.psp == 0.8
