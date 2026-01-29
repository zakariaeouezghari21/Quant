import numpy as np
import matplotlib.pyplot as plt

from core.timegrid import TimeGrid
from core.random import BrownianGenerator
from core.simulation import SimulationEngine, DiscretizationScheme
from pricing.pricers.monte_carlo.euler import EulerScheme
from tests.core.dummy_process import BrownianMotionProcess

# -----------------------------
# Paramètres
# -----------------------------
sigma = 0.5
T = 1.0
n_steps = 100
n_paths = 50_000

# -----------------------------
# Processus et simulation
# -----------------------------
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

# -----------------------------
# Histogramme
# -----------------------------
plt.hist(X_T, bins=50, density=True)
plt.title("Distribution finale X_T")
plt.xlabel("X_T")
plt.ylabel("Probability density")
plt.show()

# -----------------------------
# Courbe moyenne ± 1σ
# -----------------------------
paths_mean = paths.mean(axis=0)
paths_std = paths.std(axis=0)

plt.plot(paths_mean, label="mean")
plt.fill_between(range(len(paths_mean)), 
                 paths_mean - paths_std, paths_mean + paths_std, 
                 alpha=0.3, label="±1 std")
plt.title("Evolution moyenne ± 1σ")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()
plt.show()
