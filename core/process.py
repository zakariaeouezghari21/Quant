# core/process.py
from abc import ABC, abstractmethod
import numpy as np
from core.timegrid import TimeGrid
from core.random import BrownianGenerator

class StochasticProcess(ABC):
    """
    Represents a continuous-time stochastic process.
    """

    @abstractmethod
    def dimension(self) -> int:
        pass

    @abstractmethod
    def initial_state(self) -> np.ndarray:
        pass

    @abstractmethod
    def drift(self, t: float, state: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def diffusion(self, t: float, state: np.ndarray) -> np.ndarray:
        pass
