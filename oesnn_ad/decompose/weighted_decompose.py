import numpy as np
import numpy.typing as npt
from oesnn_ad.decompose.decompose import Decompose


class WeightedCeemdan(Decompose):

    def __init__(self, stream: npt.NDArray[np.float64],
                 trials=100,
                 range_threshold=1e-3,
                 total_power_threshold=1e-2,
                 common_ratio=0.7) -> None:
        super().__init__(stream, trials, range_threshold, total_power_threshold)
        self.common_ratio = common_ratio

    def _decompose(self, channels: int):
        return self.ceemdan.ceemdan(self.stream, max_imf=channels)

    def _merge(self, detections: npt.NDArray) -> npt.NDArray:
        if self.common_ratio < 0.5 or self.common_ratio > 1.0:
            raise ValueError("Common ratio should be in range [0.5, 1.0]")

        weights = np.array(
            [self.common_ratio**n for n in range(0, detections.shape[0])])
        weighted_detections = (
            detections.T * weights).T
        summed_columns = weighted_detections.sum(axis=0)
        return summed_columns > weights.sum() / 2
