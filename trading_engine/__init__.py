"""
Trading engine for executing trading strategies.

This package provides components for executing trading strategies
using the order management system.
"""

from .engine import TradingEngine
from .predictive_engine import PredictiveEngine

__all__ = [
    'TradingEngine',
    'PredictiveEngine'
]
