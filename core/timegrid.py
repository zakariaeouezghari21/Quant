
import numpy as np
from dataclasses import dataclass
from typing import Iterator

@dataclass(frozen=True)
class TimeGrid:
    t0: float
    t_end: float
    n_steps: int

    def __post_init__(self):
        if self.t_end <= self.t0:
            raise ValueError("t_end must be > t0")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")

    @property
    def dt(self) -> float:
        return (self.t_end - self.t0) / self.n_steps

    @property
    def times(self) -> np.ndarray:
        return np.linspace(self.t0, self.t_end, self.n_steps + 1)

    def __iter__(self) -> Iterator[float]:
        return iter(self.times)
