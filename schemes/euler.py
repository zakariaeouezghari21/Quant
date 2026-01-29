from schemes.base_scheme import DiscretizationScheme
import numpy as np

class EulerScheme(DiscretizationScheme):

    def evolve(self, process, x0, t, dt, z):
        drift = process.drift(x0, t)
        diffusion = process.diffusion(x0, t)
        return x0 + drift * dt + diffusion @ z * np.sqrt(dt)
    

