from __future__ import annotations
from typing import Protocol
from quantlib.pricing.results import PriceResult
from quantlib.pricing.product import EuropenOptions
from quantlib.pricing.model import BlackScholesModel

class Pricer(Protocol):
    def price(self, product: EuropenOptions, model: BlackScholesModel) -> PriceResult:
        ...

        
