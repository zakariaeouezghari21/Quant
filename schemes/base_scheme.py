from abc import ABC, abstractmethod

class DiscretizationScheme(ABC):

    @abstractmethod
    def evolve(self, process, x0, t, dt, z):
        pass

    