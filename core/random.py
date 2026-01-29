# core/random.py
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class BrownianGenerator:
    dim: int
    seed: int | None = None

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)

    def normal(self, size: Tuple[int, int]) -> np.ndarray:
        """
        Generate standard normal increments.
        size = (n_paths, dim)
        """
        return self.rng.standard_normal(size)

    def correlated_normals(self, corr: np.ndarray, n_paths: int) -> np.ndarray:
        """
        Generate correlated standard normals using Cholesky.
        """
        L = np.linalg.cholesky(corr)
        Z = self.normal((n_paths, self.dim))
        return Z @ L.T
