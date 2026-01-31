from __future__ import annotations
import math 
from scipy.stats import norm 

from quantlib.pricing.product import EuropenOptions, OptionType
from quantlib.pricing.model import BlackScholesModel
from quantlib.pricing.results import PriceResult

def bs_price(product: EuropenOptions, model: BlackScholesModel) -> PriceResult:
    S = model.spot
    K = product.strike
    T = product.maturity
    r = model.rate
    q = model.dividend
    sigma = model.vol

    if T <= 0:
        intrinsic = max(S - K, 0.0) if product.option_type == OptionType.CALL else max(K - S, 0.0)
        return PriceResult(price=float(intrinsic))
    
    if sigma <= 0:
        # deterministic forward
        fwd = S * math.exp((r -q) * T)
        disc = math.exp(-r * T)
        payoff = max(fwd - K, 0.0) if product.option_type == OptionType.CALL else max(K - fwd, 0.0)
        return PriceResult(price=float(disc * payoff))
    
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)

    if product.option_type == OptionType.CALL:
        price = S * disc_q * norm.cdf(d1) - K * disc_r * norm.cdf(d2)
    else:
        price = K * disc_r * norm.cdf(-d2) - S * disc_q * norm.cdf(-d1)

    return PriceResult(price=float(price), diagnostics={"d1": d1, "d2": d2})
