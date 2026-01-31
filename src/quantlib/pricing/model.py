from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class BlackScholesModel:
    spot: float
    rate: float
    dividend: float 
    vol: float

