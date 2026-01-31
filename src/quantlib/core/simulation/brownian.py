from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from quantlib.core.random.rng import RNG
from quantlib.core.time.timegrid import TimeGrid

@dataclass(frozen=True)
class BrownianMotion:
    """
    Simple Brownian motion increment genrator. 
        - Vectorized over paths
        - Supports multi-factor correlation (to do later)
    """
    rng: RNG

    def normal_increments(self, grid: TimeGrid, n_paths: int, n_factors: int=1) -> np.ndarray:
        """
        Return Z ~ N(0,1) with shape (n_paths, n_steps, n_factors).
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive.")
        if n_factors <= 0:
            raise ValueError("n_factors must be positive.")
        
        gen = self.rng.generator()
        z = gen.standard_normal(size=(n_paths, grid.n_steps, n_factors))
        return z
    
    def dW(self, grid: TimeGrid, n_paths: int, n_factors: int=1) -> np.ndarray:
        """
        Returns Brownian increments dW with shape (n_paths, n_steps, n_factors)
        where dW = sqrt(dt) * Z.
        """
        z = self.normal_increments(grid=grid, n_paths=n_paths, n_factors=n_factors)
        return  np.sqrt(grid.dt()) * z
    

    