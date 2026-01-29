import numpy as np
from core.simulation import DiscretizationScheme
from core.process import StochasticProcess

class EulerScheme(DiscretizationScheme):

    def step(self, t, dt, state, dW, process: StochasticProcess):
        drift = process.drift(t, state)
        diffusion = process.diffusion(t, state)

        return state + drift * dt + diffusion * dW
