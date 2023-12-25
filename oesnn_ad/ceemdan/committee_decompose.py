import numpy as np
import numpy.typing as npt

from oesnn_ad.ceemdan.decompose import Decompose


class CommiteeDecompose(Decompose):

    def __init__(self, stream: npt.NDArray[np.float64],
                 trials=100,
                 range_threshold=1e-3,
                 total_power_threshold=1e-2) -> None:
        super().__init__(stream, trials, range_threshold, total_power_threshold)

    def _merge(self, detections: npt.NDArray) -> npt.NDArray:
        summed_columns = detections.sum(axis=0)
        return (summed_columns / detections.shape[0]) > 0.5
