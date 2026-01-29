import numpy as np

from core.simulation import SimulationEngine
from core.timegrid import TimeGrid
from core.random import BrownianGenerator
from pricing.pricers.monte_carlo.euler import EulerScheme
from tests.core.dummy_process import BrownianMotionProcess


def test_brownian_motion_mean_variance():
    sigma = 0.5
    T = 1.0
    n_steps = 100
    n_paths = 50_000

    process = BrownianMotionProcess(sigma)
    time_grid = TimeGrid(0.0, T, n_steps)
    scheme = EulerScheme()
    brownian = BrownianGenerator(dim=1, seed=42)

    engine = SimulationEngine(
        process=process,
        time_grid=time_grid,
        scheme=scheme,
        brownian=brownian,
    )

    paths = engine.simulate(n_paths)
    X_T = paths[:, -1, 0]

    mean = X_T.mean()
    var = X_T.var()

    assert abs(mean) < 1e-2
    assert abs(var - sigma**2 * T) < 1e-2
