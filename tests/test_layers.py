"""
    Module tests Layer's methods.
"""

import numpy as np
from pytest import approx

from oesnn_ad.layer import InputLayer, OutputLayer
from oesnn_ad.neuron import OutputNeuron


WINDOW = np.array([0.5, 0.3, 0.4,
                   0.3, 0.6, 0.2,
                   1.0, 0.4, 0.3,
                   0.4, 0.2, 0.4,
                   0.1, 0.5])


def test_make_candidate():
    """
        Test asserts if candidate creation working correctly.
    """
    output_layer = OutputLayer(10)

    order, mod, c_coef, neuron_age = np.array(
        [6, 5, 3, 1, 0, 2, 4]), 0.5, 0.5, 10

    candidate = output_layer.make_candidate(
        WINDOW, order, mod, c_coef, neuron_age)

    correct_weights = np.array(
        [0.015625, 0.03125, 0.125, 0.5, 1, 0.25, 0.0625])

    assert isinstance(candidate.output_value, float)

    assert candidate.addition_time == neuron_age
    assert candidate.psp == 0
    assert candidate.modification_count == 1

    np.testing.assert_array_almost_equal(
        candidate.weights, correct_weights, decimal=3)
    assert candidate.max_psp == approx(1.333, abs=1e-3)
    assert candidate.gamma == approx(0.666, abs=1e-3)


def test_find_most_similar_without_neurons():
    """
        Test asserts if method return None and np.if when there aren't neurons in output layer.
    """
    output_layer = OutputLayer(10)
    candidate = OutputNeuron(
        np.array([0.25, 0.25, 0.25]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    neuron_result, distance = output_layer.find_most_similar(candidate)

    assert neuron_result is None
    assert np.isinf(distance)


def test_find_most_similar_with_neurons():
    """
        Test asserts if method return nearest neuron when there are neurons in output layer.
    """
    output_layer = OutputLayer(10)

    neuron_out1 = OutputNeuron(
        np.array([0.26, 0.26, 0.26]), 0.25, 0.1, 1, 0.25, 0.75, 2)
    neuron_out2 = OutputNeuron(
        np.array([1.0, 1.0, 1.0]), 0.25, 0.1, 1, 0.25, 0.75, 2)
    neuron_out3 = OutputNeuron(
        np.array([0.0, 0.0, 0.0]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    output_layer.add_new_neuron(neuron_out1)
    output_layer.add_new_neuron(neuron_out2)
    output_layer.add_new_neuron(neuron_out3)

    c_neuron = OutputNeuron(
        np.array([0.25, 0.25, 0.25]), 0.25, 0.1, 1, 0.25, 0.75, 2)

    neuron_result, distance = output_layer.find_most_similar(c_neuron)

    assert neuron_result == neuron_out1
    assert distance == approx(0.0173, abs=1e-4)


def test_reset_psp():
    """
        Test assert if zeroing PSP working correctly.
    """
    output_layer = OutputLayer(10)
    neuron_out1 = OutputNeuron(
        np.array([]), 0.0, 0.0, 0.0, 0.0, PSP=5.0, max_PSP=10.0)
    neuron_out2 = OutputNeuron(
        np.array([]), 0.0, 0.0, 0.0, 0.0, PSP=6.0, max_PSP=10.0)
    neuron_out3 = OutputNeuron(
        np.array([]), 0.0, 0.0, 0.0, 0.0, PSP=7.0, max_PSP=10.0)

    output_layer.add_new_neuron(neuron_out1)
    output_layer.add_new_neuron(neuron_out2)
    output_layer.add_new_neuron(neuron_out3)

    output_layer.reset_psp()

    for neuron in output_layer:
        assert neuron.psp == 0.0


def test_add_new_neuron():
    """
        Test assert if method add new neuron correctly.
    """
    output_layer = OutputLayer(10)
    assert output_layer.num_neurons == 0
    assert len(output_layer) == 0

    neuron_out1 = OutputNeuron(np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    output_layer.add_new_neuron(neuron_out1)
    assert output_layer.num_neurons == 1
    assert len(output_layer) == 1
    assert output_layer[0] == neuron_out1

    neuron_out2 = OutputNeuron(np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    output_layer.add_new_neuron(neuron_out2)
    assert output_layer.num_neurons == 2
    assert len(output_layer) == 2
    assert output_layer[1] == neuron_out2


def test_replace_oldest():
    """
        Test assert if method replace oldest neuron correctly.
    """
    output_layer = OutputLayer(10)

    neuron_out1 = OutputNeuron(np.array([]), 0.0, 0.0, 0.0,
                               PSP=0.0, max_PSP=0.0, addition_time=1.0)
    neuron_out2 = OutputNeuron(np.array([]), 0.0, 0.0, 0.0,
                               PSP=0.0, max_PSP=0.0, addition_time=2.0)
    neuron_out3 = OutputNeuron(np.array([]), 0.0, 0.0, 0.0,
                               PSP=0.0, max_PSP=0.0, addition_time=3.0)
    candidate = output_layer.make_candidate(
        WINDOW, np.array([1, 2, 3]), 0.0, 0.0, 10)

    output_layer.add_new_neuron(neuron_out1)
    output_layer.add_new_neuron(neuron_out2)
    output_layer.add_new_neuron(neuron_out3)

    output_layer.replace_oldest(candidate)

    assert neuron_out1 not in output_layer
    assert candidate in output_layer


def test_len_magic_method():
    """
        Test assert if len method working correctly for Layer's classes.
    """
    input_layer = InputLayer(10)

    assert len(input_layer) == 10

    output_layer = OutputLayer(10)

    assert len(output_layer) == 0

    output_layer.add_new_neuron(OutputNeuron(
        np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    assert len(output_layer) == 1
