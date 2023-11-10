"""
    Module contains definition and implementation of layer class.
"""

from typing import Generator, List, Tuple

import numpy as np
import numpy.typing as npt

from grf_init import GRFInit
from neuron import InputNeuron, Neuron, OutputNeuron


class Layer:
    """
        Base class creates interface for inheritance classes.
        Beside of common attributes for all layers, class is making magic methods allowing
        indexation, counting numer of neurons in layer thank to 'len' function.
        Class musn't be created as object.
    """

    def __init__(self, num_neurons: int) -> None:
        """
            Args:
                num_neurons (int): Number of neurons in layer
        """
        self.num_neurons = num_neurons

        self.neurons: List[Neuron]

    def __iter__(self) -> Generator[Neuron, None, None]:
        """
            Magic method for iteration over list of neurons in layer.
            Yields:
                Generator[Neuron, None, None]: generator which allow to iterate
                over object's neurons
        """
        return (neuron for neuron in self.neurons)

    def __len__(self) -> int:
        """
            Magic method returning count of neurons in layer.

            Returns:
                int: count of neurons in layer
        """
        return len(self.neurons)

    def __getitem__(self, index: int) -> Neuron:
        """
            Magic method allowing to get neuron with usage of indexation.

            Args:
                index (int): index of neuron in list

            Returns:
                Neuron: neuron under index in list
        """
        return self.neurons[index]


class InputLayer(Layer):
    """
        Class implementing input layer, inheriting after base class Layer.
        Class stores and handling list of input neurons.
    """

    def __init__(self, input_size: int) -> None:
        """
            Args:
                input_size (int): number of neurons in input layer
        """
        super().__init__(input_size)

        self.neurons: List[InputNeuron] = [
            InputNeuron(0.0, id) for id in range(input_size)]

    def __iter__(self) -> Generator[InputNeuron, None, None]:
        """
            Magic method for iteration over list of input neurons in layer.
            Yields:
                Generator[InputNeuron, None, None]: generator which allow to iterate
                over input neurons
        """
        neurons = sorted(self.neurons, key=lambda neuron: neuron.order)
        return (neuron for neuron in neurons)

    def __getitem__(self, index: int) -> InputNeuron:
        """
            Magic method allowing to get input neuron with usage of indexation.

            Args:
                index (int): index of input neuron in list

            Returns:
                InputNeuron: input neuron under index in list
        """
        return super().__getitem__(index)

    @property
    def orders(self) -> np.vectorize:
        """
            Property, which lookup firing order of neurons in one list
            Returns:
                np.vectorize: vectorized list of firing order of neurons
        """
        vectorized_get_order = np.vectorize(lambda neuron: neuron.order)
        return vectorized_get_order(self.neurons)

    def set_orders(self,
                   window: npt.NDArray[np.float64],
                   ts_coef: float,
                   mod: float,
                   beta: float) -> None:
        """
            Method set for all input neurons new firing order in layer.
            Args:
                window (npt.NDArray[np.float64]): list of input values from stream
                ts_coef (float): factor from OeSNN-AD
                mod (float): factor from OeSNN-AD
                beta (float): factor from OeSNN-AD
        """
        grf = GRFInit(window, self.num_neurons, ts_coef, mod, beta)

        for neuron, new_order in zip(self.neurons, grf.get_order()):
            neuron.set_order(new_order)


class OutputLayer(Layer):
    """
        Class implementing output layer, inheriting after base class Layer.
        Class stores and handling list of output neurons.
    """

    def __init__(self, max_output_size: int) -> None:
        """
            Args:
                max_output_size (int): Max number of output neurons in layer
        """

        self.max_outpt_size = max_output_size
        self.neurons: List[OutputNeuron] = []

    def __iter__(self) -> Generator[OutputNeuron, None, None]:
        """
            Magic method for iteration over list of output neurons in layer.
            Yields:
                Generator[InputNeuron, None, None]: generator which allow to iterate
                over output neurons
        """
        return super().__iter__()

    def __getitem__(self, index: int) -> OutputNeuron:
        """
            Magic method allowing to get output neuron with usage of indexation.

            Args:
                index (int): index of output neuron in list

            Returns:
                OutputNeuron: output neuron under index in list
        """
        return super().__getitem__(index)

    @property
    def num_neurons(self):
        return len(self.neurons)

    def make_candidate(self,
                       window: npt.NDArray[np.float64],
                       order: npt.NDArray[np.intp],
                       mod: float,
                       c_coef: float,
                       neuron_age: int) -> OutputNeuron:
        """
            Method is making new output neuron and setting his properties
            Args:
                window (npt.NDArray[np.float64]): list of values from stream
                order (npt.NDArray[np.intp]): firing order of neuron
                mod (float): factor from OeSNN-AD
                c_coef (float): factor from OeSNN-AD
                neuron_age (int): neuron's age
            Returns:
                OutputNeuron: Candidate neuron
        """
        weights = mod ** order
        output_value = np.random.normal(np.mean(window), np.std(window))
        psp_max = (weights * (mod ** order)).sum()
        gamma = c_coef * psp_max

        return OutputNeuron(weights, gamma,
                            output_value, 1, neuron_age,
                            0, psp_max)

    def find_most_similar(self,
                          candidate_neuron: OutputNeuron) -> Tuple[OutputNeuron | None, float]:
        """
            Method return neuron which have lowest euclidean distance to candidate neuron and that
            distance. If layer doesn't have neurons, method return Tuple[false, np.inf]
            Args:
                candidate_neuron (OutputNeuron): Neuron for which we need to find most similar
            Returns:
                Tuple[OutputNeuron | bool, float]: Two elements tuple, in which first position is
                for neuron or boolean false if layer is empty and second position is for
                euclidean distance (np.inf if layer is empty)
        """
        if not self.neurons:
            return None, np.Inf

        def dist_f(neuron: OutputNeuron):
            return np.linalg.norm(neuron.weights - candidate_neuron.weights)

        most_similar_neuron = min(self.neurons, key=dist_f)
        min_distance = dist_f(most_similar_neuron)
        return most_similar_neuron, min_distance

    def add_new_neuron(self, neuron: OutputNeuron) -> None:
        """
            Method add new neuron when number of neurons is lower than max size of layer.
            Additionaly method after pushinh new neuron, update value of attribute num_neurons.
            Args:
                neuron (OutputNeuron): New neuron in layer
        """
        self.neurons.append(neuron)

    def replace_oldest(self, candidate: OutputNeuron) -> None:
        """
            Method replace oldest neuron in layer by new created candidate, when number
            of neurons in layer is max.
            Args:
                candidate (OutputNeuron): new neuron which replace oldest neuron in layer
        """
        oldest = min(self.neurons, key=lambda n: n.addition_time)
        self.neurons.remove(oldest)
        self.neurons.append(candidate)

    def reset_psp(self) -> None:
        """
            Method zeroing postsynaptic potential all neurons in layer
        """
        for neuron in self.neurons:
            neuron.psp = 0
