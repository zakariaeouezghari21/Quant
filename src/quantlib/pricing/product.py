from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"

@dataclass(frozen=True)
class EuropenOptions:
    strike: float
    maturity: float   #in years
    option_type: OptionType = OptionType.CALL

    