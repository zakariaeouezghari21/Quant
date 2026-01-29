# core/market.py
from dataclasses import dataclass
from typing import Callable

@dataclass
class YieldCurve:
    discount_factor: Callable[[float], float]

    def df(self, t: float) -> float:
        return self.discount_factor(t)


@dataclass
class MarketData:
    spot: float
    rate_curve: YieldCurve
    dividend_curve: YieldCurve | None = None

    def forward(self, t: float) -> float:
        r_df = self.rate_curve.df(t)
        q_df = self.dividend_curve.df(t) if self.dividend_curve else 1.0
        return self.spot * q_df / r_df
