"""
    Module contains definition and implementation of class GRFInit.
"""

import numpy as np
import numpy.typing as npt


class GRFInit:
    """
        Class contains functions which target is initialization of firing neurons order
        in output layer.
        Class's object is redefined in every iteration of OeSNN-AD.
    """

    def __init__(self, window: npt.NDArray[np.float64], input_size: int, ts_factor: float,
                 mod: int, beta: float) -> None:
        """
            Args:
                window (npt.NDArray[np.float64]): window of values in current iteration
                input_size (int): number of neurons in input layer
                ts_factor (float): factor from OeSNN-AD
                mod (int): factor from OeSNN-AD
                beta (float): factor from OeSNN-AD
        """
        self.min_w_i: float = window.min()
        self.max_w_i: float = window.max()
        self.window_head: float = window[-1]

        self.input_size = input_size
        self.ts_factor = ts_factor
        self.mod = mod
        self.beta = beta

    def _get_width_vec(self) -> npt.NDArray[np.float64]:
        """
            Method to calculate Gaussian Receptor Field width vector for all neurons in input layer.
            If GRF value value is equal 0, it will be replaced to 1.
            Returns:
                npt.NDArray[np.float64]: GRF's width vector
        """
        value = (self.max_w_i - self.min_w_i) / \
            ((self.input_size - 2) * self.beta)
        if value == 0.0:
            value = 1.0
        return np.repeat(value, self.input_size)

    def _get_center_vec(self) -> npt.NDArray[np.float64]:
        """
            Method to calculate Gaussian Receptor Field center vector for all neurons
            in input layer.
            Returns:
                npt.NDArray[np.float64]: GRF's center vector
        """
        return (self.min_w_i + ((2*np.arange(0, self.input_size, 1) - 3) / 2) *
                (self.max_w_i - self.min_w_i) / (self.input_size - 2))

    def _get_excitation(self,
                        width_v: npt.NDArray[np.float64],
                        center_v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
            Method to calculate Gaussian Receptor Field excitation vector for all neurons
            in input layer.
            Args:
                width_v (npt.NDArray[np.float64]): GRF's width vector
                center_v (npt.NDArray[np.float64]): GRF's center vector
            Returns:
                npt.NDArray[np.float64]: GRF's excitation vector
        """
        rep_xt = np.repeat(self.window_head, self.input_size)
        return np.exp(-0.5 * ((rep_xt - center_v) / width_v) ** 2)

    def _get_firing_time(self, excitation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
            Method to calculate Gaussian Receptor Field firing times vector for all neurons
            in input layer.
            Args:
                excitation (npt.NDArray[np.float64]): GRF's excitation vector
            Returns:
                npt.NDArray[np.float64]: firing times vector
        """
        return self.ts_factor * (np.ones(self.input_size) - excitation)

    def _get_order(self, firings_times: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
        """
            Method to calculate Gaussian Receptor Field firing order vector for all neurons
            in input layer.
            Args:
                firings_times (npt.NDArray[np.float64]): firing times vector
            Returns:
                npt.NDArray[np.intp]: orders vector
        """
        arg_sorted = np.argsort(firings_times)
        orders = np.empty_like(arg_sorted)
        orders[arg_sorted] = np.arange(len(firings_times))
        return orders

    def get_order(self) -> npt.NDArray[np.intp]:
        """
            Method is public interface to calculate vector of firing's orders for all neurons
            in input layer.
            Returns:
                npt.NDArray[np.intp]: orders vector
        """
        width_v = self._get_width_vec()
        center_v = self._get_center_vec()
        excitation = self._get_excitation(width_v, center_v)
        firings_times = self._get_firing_time(excitation)

        return self._get_order(firings_times)
