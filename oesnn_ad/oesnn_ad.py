"""
    Module contains main class of alghorithm, which is main interface of model.
"""

from typing import List

import numpy as np
import numpy.typing as npt

from layer import InputLayer, OutputLayer
from neuron import OutputNeuron


class OeSNNAD:
    """
        Class implementing OeSNN-AD model. Data stream is passed as parameter in
        constructor, with all hyperparameters. Main interface of class is method
        predict, which return vector with predictions.
    """

    def __init__(self, stream: npt.NDArray[np.float64], window_size: int = 100,
                 num_in_neurons: int = 10, num_out_neurons: int = 50,
                 ts_factor: float = 1000.0, mod: float = 0.6, c_factor: float = 0.6,
                 epsilon: float = 2, ksi: float = 0.9, sim: float = 0.15,
                 beta: float = 1.6) -> None:
        """

        Args:
            stream (npt.NDArray[np.float64]): data stream from dataset
            window_size (int): size of data window. Defaults to 100.
            num_in_neurons (int): input neuron's number. Defaults to 10.
            num_out_neurons (int): output neuron's number. Defaults to 50.
            ts_factor (float): OeSNN-AD specific factor. Defaults to 1000.0.
            mod (float): OeSNN-AD specific factor. Defaults to 0.6.
            c_factor (float): OeSNN-AD specific factor. Defaults to 0.6.
            epsilon (float): OeSNN-AD specific factor. Defaults to 2.
            ksi (float): OeSNN-AD specific factor. Defaults to 0.9.
            sim (float): OeSNN-AD specific factor. Defaults to 0.15.
            beta (float): OeSNN-AD specific factor. Defaults to 1.6.
        """
        self.stream = stream
        self.stream_len = self.stream.shape[0]
        self.window_size = window_size

        self.input_layer: InputLayer = InputLayer(num_in_neurons)
        self.output_layer: OutputLayer = OutputLayer(num_out_neurons)

        self.ts_factor = ts_factor
        self.mod = mod
        self.c_factor = c_factor

        self.gamma = self.c_factor * \
            (1 - self.mod**(2*num_in_neurons)) / (1 - self.mod**2)
        self.epsilon = epsilon
        self.ksi = ksi
        self.sim = sim
        self.beta = beta

        self.values: List[float] = []
        self.anomalies: List[bool] = []
        self.errors: List[float] = []

    def _get_window_from_stream(self, begin_idx: int,
                                end_idx: int) -> npt.NDArray[np.float64]:
        """
            Method returning window with data.

            Args:
                begin_idx (int): begin index of data window
                end_idx (int): end index of data window
                
            Returns:
                npt.NDArray[np.float64]: data window
        """
        return self.stream[begin_idx: end_idx]

    def _init_values_rand(self, window: npt.NDArray[np.float64]) -> List[float]:
        """
            Method which init values in attribute on begining.

            Args:
                window (npt.NDArray): _description_
                
            Returns:
                List[float]: _description_
        """
        return np.random.normal(
            np.mean(window), np.std(window), self.window_size).tolist()

    def _init_new_arrays_for_predict(self, window: npt.NDArray[np.float64]) -> None:
        """
            Method initilize attributes like values, errors and anomaliesl
            
            Args:
                window (npt.NDArray[np.float64]): window from datastream
        """
        self.values = self._init_values_rand(window)
        self.errors = [np.abs(xt - yt) for xt, yt in zip(window, self.values)]
        self.anomalies = [False] * self.window_size

    def predict(self) -> npt.NDArray[np.bool_]:
        """
            Method is main interface of model. There is main flow of model, with returning
            vector of predictions.
            
            Returns:
                npt.NDArray[np.bool_]: predictions vector
        """
        window = self._get_window_from_stream(0, self.window_size)

        self._init_new_arrays_for_predict(window)
        for age in range(self.window_size + 1, self.stream_len):
            self.input_layer.set_orders(
                window, self.ts_factor, self.mod, self.beta)

            window = self._get_window_from_stream(age - self.window_size, age)

            self._learning(window, age)

            self._anomaly_detection(window)

        return np.array(self.anomalies)

    def _anomaly_detection(self, window: npt.NDArray[np.float64]) -> None:
        """
            Method check if anomaly is detected.
            
            Args:
                window (npt.NDArray[np.float64]): window from datastream
        """
        window_head = window[-1]
        first_fired_neuron = self._fires_first()
        if first_fired_neuron:
            self.values.append(first_fired_neuron.output_value)
            self.errors.append(first_fired_neuron.error_calc(window_head))
            self.anomalies.append(self._anomaly_classification())
        else:
            self.values.append(None)
            self.errors.append(np.abs(window_head))
            self.anomalies.append(True)

    def _anomaly_classification(self) -> bool:
        """
            Method check if current head of window is anomaly.
            
            Returns:
                bool: _description_
        """
        error_t = self.errors[-1]
        errors_window = np.array(self.errors[-(self.window_size):-1])
        anomalies_window = np.array(self.anomalies[-(self.window_size - 1):])

        errors_for_non_anomalies = errors_window[np.where(~anomalies_window)]
        return not (
            (not np.any(errors_for_non_anomalies)) or (error_t - np.mean(errors_for_non_anomalies)
                                                < np.std(errors_for_non_anomalies) * self.epsilon)
        )

    def _learning(self, window: npt.NDArray[np.float64], neuron_age: int) -> None:
        """
            Method learn model by tune up parameters of output neurons.

            Args:
                window (npt.NDArray[np.float64]): window from data stream
                neuron_age (int): number of current iteration
        """
        anomaly_t, window_head = self.anomalies[-1], window[-1]
        candidate_neuron = self.output_layer.make_candidate(window, self.input_layer.orders,
                                                            self.mod, self.c_factor, neuron_age)

        if not anomaly_t:
            candidate_neuron.error_correction(window_head, self.ksi)

        most_familiar_neuron, dist = self.output_layer.find_most_similar(
            candidate_neuron)

        if dist <= self.sim:
            most_familiar_neuron.update_neuron(candidate_neuron)
        elif self.output_layer.num_neurons < self.output_layer.max_outpt_size:
            self.output_layer.add_new_neuron(candidate_neuron)
        else:
            self.output_layer.replace_oldest(candidate_neuron)

    def _fires_first(self) -> OutputNeuron | None:
        """
            Method control PSP in model, and returning first firing output neuron (if fired
            more than one output neuron, there is chosen with greatest PSP)
            
            Returns:
                OutputNeuron | bool: firing neuron with greatest PSP 
        """
        self.output_layer.reset_psp()

        for neuron_input in self.input_layer:
            fired_neuron = None
            for n_out in self.output_layer:
                n_out.update_psp(n_out[neuron_input.neuron_id] * (self.mod ** neuron_input.order))

                if n_out.psp > self.gamma:
                    fired_neuron = n_out

            if fired_neuron is not None:
                return fired_neuron
            
        return None
