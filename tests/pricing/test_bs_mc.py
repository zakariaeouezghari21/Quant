from quantlib.pricing.product import EuropenOptions, OptionType
from quantlib.pricing.model import BlackScholesModel
from quantlib.pricing.pricers.analytic.black_scholes_closed_form import bs_price
from quantlib.pricing.pricers.monte_carlo.mc_engine import bs_mc_price, MonteCarloSettings

def test_bs_mc_close_to_closed_form():
    model = BlackScholesModel(spot=100.0, rate=0.02, dividend=0.0, vol=0.2)
    opt = EuropenOptions(strike=100.0, maturity=1.0, option_type=OptionType.CALL)

    cf = bs_price(opt, model).price

    settings = MonteCarloSettings(n_paths=120_000, n_steps=252, seed=123, antithetic=True)
    mc = bs_mc_price(opt, model, settings=settings)

    # tolerance depends on paths; with antithetic 120k paths should be pretty stable
    assert abs(mc.price - cf) < 0.25

    # ensure closed form is inside MC 95% CI most of the time    
    lo , hi = mc.diagnostics["ci95"]
    assert lo < cf < hi

def test_bs_mc_reproducible_with_seed():
    model = BlackScholesModel(spot=100.0, rate=0.01, dividend=0.0, vol=0.25)
    opt = EuropenOptions(strike=110.0, maturity=0.5, option_type=OptionType.PUT)

    settings = MonteCarloSettings(n_paths=50_000, n_steps=126, seed=999, antithetic=True)
    mc1 = bs_mc_price(opt, model, settings=settings).price
    mc2 = bs_mc_price(opt, model, settings=settings).price

    assert mc1 == mc2