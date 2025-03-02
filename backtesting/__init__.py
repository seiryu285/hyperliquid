"""
Backtesting package for evaluating trading strategies.

This package provides tools for backtesting trading strategies
using historical price data.
"""

from backtesting.engine import BacktestEngine
from backtesting.strategy import (
    BaseStrategy, 
    TrendFollowingStrategy, 
    MeanReversionStrategy,
    PredictiveModelStrategy
)
from backtesting.utils import (
    run_backtest, 
    plot_equity_curve, 
    plot_drawdown, 
    plot_trade_analysis, 
    print_statistics,
    optimize_strategy_parameters
)

__all__ = [
    'BacktestEngine',
    'BaseStrategy',
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'PredictiveModelStrategy',
    'run_backtest',
    'plot_equity_curve',
    'plot_drawdown',
    'plot_trade_analysis',
    'print_statistics',
    'optimize_strategy_parameters'
]
