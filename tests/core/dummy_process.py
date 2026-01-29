# On test dX_t = sigma * dW_t
# On connait la solution X_T = N(0, sigma**2 * T)

import numpy as np
from core.process import StochasticProcess

class BrownianMotionProcess(StochasticProcess):
    def __init__(self, sigma: float):
        self.sigma = sigma

    def dimension(self) -> int:
        return 1

    def initial_state(self) -> np.ndarray:
        return np.array([0.0])

    def drift(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.array([0.0])

    def diffusion(self, t: float, state: np.ndarray) -> np.ndarray:
        return np.array([self.sigma])
