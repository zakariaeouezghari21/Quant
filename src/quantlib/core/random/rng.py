from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.random import Generator, PCG64

@dataclass(frozen=True)
class RNG:
    seed: int = 12345

    def generator(self) -> Generator:
        return Generator(PCG64(self.seed))
    
