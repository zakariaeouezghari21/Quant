from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from quantlib.core.random.rng import RNG
from quantlib.core.time.timegrid import TimeGrid
from quantlib.core.simulation.brownian import BrownianMotion
from quantlib.pricing.product import EuropenOptions, OptionType
from quantlib.pricing.model import BlackScholesModel
from quantlib.pricing.results import PriceResult

@dataclass(frozen=True)
class MonteCarloSettings:
    n_paths: int = 200_000
    n_steps: int = 252
    seed: int = 12345
    antithetic: bool = True

def _payoff_european(option: EuropenOptions, spot_T: np.ndarray) -> np.ndarray:
    if option.option_type == OptionType.CALL:
        return np.maximum(spot_T - option.strike, 0.0)
    return np.maximum(option.strike - spot_T, 0.0)

def bs_mc_price(product: EuropenOptions, model: BlackScholesModel, settings: MonteCarloSettings=MonteCarloSettings()) -> PriceResult:
    """
    MC pricing under BS with Euler exact discretization for log(S):
        dS/S = (r - q) dt + sigma dW
    Exact step:
        S_{t+dt} = S_t * exp((r-q-0.5*sigma^2)dt + sigma*sqrt(dt)*Z)
    Returns price + diagnostics (stderr, CI, mean payoff, etc).
    """
    if product.maturity <= 0:
        intrinsic = max(model.spot - product.strike, 0.0) if product.option_type == OptionType.CALL else max(product.strike - model.spot, 0.0)
        return PriceResult(price=float(intrinsic), diagnostics={"n_paths": 0, "n_steps":0})
    
    n_paths = int(settings.n_paths)
    n_steps = int(settings.n_steps)
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive.")
    
    grid = TimeGrid(t0=0.0, t1=float(product.maturity), n_steps=n_steps)
    dt = grid.dt()

    r = float(model.rate)
    q = float(model.dividend)
    sigma = float(model.vol)
    S0 = float(model.spot)

    disc = math.exp(-r * product.maturity)
    drift = (r - q - 0.5 * sigma* sigma) * dt
    vol_step = sigma * math.sqrt(dt)

    rng = RNG(seed=settings.seed)
    bm = BrownianMotion(rng=rng)

    # Antithetic variates: generate half paths and mirror
    if settings.antithetic:
        n_half = (n_paths + 1) // 2
        z = bm.normal_increments(grid=grid, n_paths=n_half, n_factors=1).squeeze(-1)  # (n_half, n_steps)
        z_full = np.vstack([z, -z])[:n_paths]  # (n_paths, n_steps)
    else:
        z_full = bm.normal_increments(grid=grid, n_paths=n_paths, n_factors=1).squeeze(-1)

    # simulate terminal spot vectorized (exact log scheme)
    logS = math.log(S0) + np.cumsum(drift + vol_step * z_full, axis=1)  # (n_paths, n_steps)
    ST = np.exp(logS[:, -1])

    payoffs = _payoff_european(product, ST)
    discounted = disc * payoffs

    price = float(np.mean(discounted))

    # standard error + 95% CI (normal approx)
    # Use ddof=1 for unbiased sample variance 
    sample_std = float(np.std(discounted, ddof=1))
    stderr = sample_std / math.sqrt(n_paths)
    ci95_low = price - 1.96 * stderr
    ci95_high = price + 1.96 * stderr

    return PriceResult(price=price, diagnostics={"n_paths": n_paths, "n_steps": n_steps, "seed": settings.seed, "antithetic": settings.antithetic, "stderr": stderr, "ci95": (ci95_low, ci95_high), "mean_payoff": float(np.mean(payoffs))})

