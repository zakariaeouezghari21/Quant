# core/simulation.py
from abc import ABC, abstractmethod
import numpy as np

from core.process import StochasticProcess
from core.timegrid import TimeGrid
from core.random import BrownianGenerator


# ============================================================
# Discretization scheme interface
# ============================================================

class DiscretizationScheme(ABC):
    """
    Time discretization scheme for SDEs.
    """

    @abstractmethod
    def step(
        self,
        t: float,
        dt: float,
        state: np.ndarray,
        dW: np.ndarray,
        process: StochasticProcess
    ) -> np.ndarray:
        pass


# ============================================================
# Simulation Engine
# ============================================================

class SimulationEngine:
    """
    Generic Monte Carlo simulation engine.
    """

    def __init__(
        self,
        process: StochasticProcess,
        time_grid: TimeGrid,
        scheme: DiscretizationScheme,
        brownian: BrownianGenerator,
    ):
        self.process = process
        self.time_grid = time_grid
        self.scheme = scheme
        self.brownian = brownian

        if self.brownian.dim != process.dimension():
            raise ValueError("Brownian dimension does not match process dimension")

    def simulate(self, n_paths: int) -> np.ndarray:
        """
        Simulate paths of the stochastic process.

        Returns:
            paths: shape (n_paths, n_steps + 1, dim)
        """
        times = self.time_grid.times
        dt = self.time_grid.dt
        dim = self.process.dimension()

        paths = np.zeros((n_paths, len(times), dim))
        paths[:, 0, :] = self.process.initial_state()

        for i in range(1, len(times)):
            t = times[i - 1]

            dW = np.sqrt(dt) * self.brownian.normal((n_paths, dim))

            paths[:, i, :] = self.scheme.step(
                t=t,
                dt=dt,
                state=paths[:, i - 1, :],
                dW=dW,
                process=self.process,
            )

        return paths
