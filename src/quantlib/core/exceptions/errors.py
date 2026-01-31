class QuantLibError(Exception):
    """Base exception for the library."""

class MarketDataError(QuantLibError):
    pass

class PricingError(QuantLibError):
    pass

class CalibrationError(QuantLibError):
    pass

