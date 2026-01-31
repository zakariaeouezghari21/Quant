from quantlib.pricing.product import EuropenOptions, OptionType
from quantlib.pricing.model import BlackScholesModel
from quantlib.pricing.pricers.analytic.black_scholes_closed_form import bs_price

import math

def test_bs_call_price_smoke():
    model = BlackScholesModel(spot=100.0, rate=0.05, dividend=0.0, vol=0.2)
    opt = EuropenOptions(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
    res = bs_price(opt, model)
    assert 9.0 < res.price < 12.0  # 10.45

def test_bs_put_call_parity():
    model = BlackScholesModel(spot=100.0, rate=0.01, dividend=0.0, vol=0.25)
    K, T = 110.0, 0.5
    call = EuropenOptions(strike=K, maturity=T, option_type=OptionType.CALL)
    put = EuropenOptions(strike=K, maturity=T, option_type=OptionType.PUT)

    c = bs_price(call, model).price
    p = bs_price(put, model).price

    # put-call parity: c - p = S - K*exp(-rT)  (q=0)
    rhs = model.spot - K * math.exp(-model.rate * T)
    assert abs((c - p) - rhs) < 1e-6
