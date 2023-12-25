from threading import Thread

import numpy as np
import numpy.typing as npt
from PyEMD import CEEMDAN
from oesnn_ad import OeSNNAD


class Decompose:
    def __init__(self, stream: npt.NDArray[np.float64],
                 trials=100,
                 range_threshold=1e-3,
                 total_power_threshold=1e-2) -> None:
        self.stream = stream
        self.ceemdan = CEEMDAN(trials=trials, range_threshold=range_threshold,
                               total_power_threshold=total_power_threshold)

    def _decompose(self, channels: int):
        return self.ceemdan.ceemdan(self.stream, max_imf=channels)

    def _merge(self, detections: npt.NDArray) -> npt.NDArray:
        raise NotImplementedError

    def predict(self, cemmndan_channels: int = -1, **parameters):
        channels = list(self._decompose(cemmndan_channels))
        channels.insert(0, self.stream)
        detections = []
        threads = [Thread(
            target=detections.append(
                (idx, OeSNNAD(component, **parameters).predict()))
        ) for idx, component in enumerate(channels)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        detections = [x[1] for x in sorted(detections)]
        return self._merge(np.array(detections))
