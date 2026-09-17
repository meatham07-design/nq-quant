"""Compatibility shim for the upstream V4 audit branch.

The committed experiments/unified_engine_1m.py imports UnifiedZone from this
module, but the upstream repository omitted the module itself. This file
restores only the data container required by that engine; it does not change
signal, execution, sizing, risk, or exit logic.
"""
from dataclasses import dataclass

@dataclass
class UnifiedZone:
    direction: int
    top: float
    bottom: float
    size: float
    birth_bar: int
    birth_atr: float
    tier: int
    sweep_range_atr: float = 0.0
    used: bool = False
