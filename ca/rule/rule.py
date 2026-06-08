from abc import ABC, abstractmethod

import numpy as np

class Rule(ABC):
    @abstractmethod
    def apply(self, grid: np.ndarray) -> np.ndarray:
        raise NotImplementedError()