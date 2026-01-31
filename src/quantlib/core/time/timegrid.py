from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class TimeGrid:
    t0: float
    t1: float
    n_steps: int

    def times(self) -> np.ndarray:
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive.")
        return np.linspace(self.t0, self.t1, self.n_steps + 1)
    
    def dt(self) -> float:
        return (self.t1 - self.t0) / self.n_steps
    
    