"""
    Module contains definitions and neuron's implementation.
"""
import numpy as np
import numpy.typing as npt


class Neuron:
    """
        Base class of neuron
    """

    def __init__(self) -> None:
        pass

class InputNeuron(Neuron):
    """
        Input neuron class inheriting after neuron's base class.
    """

    def __init__(self, firing_time: float, neuron_id: int = 0, order: int = 0) -> None:
        """
            Args:
                firing_time (float): Firing time of neuron
                neuron_id (int, optional): Neuron identificator. Defaults to 0.
                order (int, optional): Order of neuron firing in layer. Defaults to 0.
        """
        super().__init__()
        self.neuron_id = neuron_id
        self.firing_time = firing_time
        self.order = order

    def set_order(self, new_order: int):
        """
            Setter for order attribute.
            
            Args:
                new_order (int): New neuron order
        """
        self.order = new_order

class OutputNeuron(Neuron):
    """
        Output neuron class, inheriting after neuron's base class.
    """

    def __init__(self, weights: npt.NDArray[np.float64], gamma: float,
                 output_value: float, modification_count: float, addition_time: float,
                 PSP: float, max_PSP: float) -> None:
        """
            Args:
                weights (npt.NDArray[np.float64]): numpy array of neuron's weights
                gamma (float): factor specific for OeSNN-AD
                output_value (float): Output value of output neuron
                modification_count (float): count of neuron modifications
                addition_time (float): Update time of output neuron
                PSP (float): postsynaptic potential
                max_PSP (float): maximal postsynaptic potential
        """
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        self.output_value = output_value
        self.modification_count = modification_count
        self.addition_time = addition_time
        self.psp = PSP
        self.max_psp = max_PSP

    def __getitem__(self, index: int) -> float:
        """
            Magic method which return value of weight under specific index.

            Args:
                index (int): specific index

            Returns:
                float: value of weight under index
        """
        return self.weights[index]

    def update_neuron(self, candidate_neuron: 'OutputNeuron') -> None:
        """
            Method to update properties of neuron based on other neuron which is argument
            of method.
            
            Args:
                candidate_neuron (OutputNeuron): Candidate neuron which properties will be 
                base of neuron modification
        """
        self.weights = (candidate_neuron.weights +
                        self.modification_count * self.weights) / (self.modification_count + 1)
        self.output_value = ((candidate_neuron.output_value +
                             self.modification_count * self.output_value) /
                             (self.modification_count + 1))
        self.addition_time = ((
            candidate_neuron.addition_time + self.modification_count * self.addition_time) /
            (self.modification_count + 1))
        self.modification_count += 1

    def error_correction(self, window_head: float, ksi: float) -> None:
        """
            Correction of output value based on current head of window and
            ksi factor.

            Args:
                window_head (float): header value in window
                ksi (float): factor specific for oeSNN-AD
        """
        self.output_value += (window_head - self.output_value) * ksi

    def error_calc(self, window_head: float) -> float:
        """
            Calculation of error between value of neuron and header of window.

            Args:
                window_head (float): header value in window

            Returns:
                float: value of error
        """
        return np.abs(window_head - self.output_value)

    def update_psp(self, psp_update : float):
        """
            Method to update postsynaptic potential.

            Args:
                psp_update (float): quantity of post synaptic potential to update
        """
        self.psp += psp_update
