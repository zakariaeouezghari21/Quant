from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass(frozen=True)
class PriceResult:
    price: float
    greeks: Dict[str, float] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    