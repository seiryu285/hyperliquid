"""
Utility functions for backtesting.

This module provides utility functions for running backtests,
visualizing results, and optimizing strategy parameters.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Tuple, Optional, Callable
from datetime import datetime

from backtesting.engine import BacktestEngine
from backtesting.strategy import BaseStrategy

# Configure logging
logger = logging.getLogger(__name__)


def run_backtest(data: pd.DataFrame, 
                strategy: BaseStrategy,
                initial_capital: float = 10000.0,
                commission: float = 0.0005,
                slippage: float = 0.0001) -> Dict[str, Any]:
    """
    Run a backtest using the provided data and strategy.
    
    Args:
        data: DataFrame with price data
        strategy: Strategy instance
        initial_capital: Initial capital for the backtest
        commission: Commission rate per trade
        slippage: Slippage rate per trade
        
    Returns:
        Dictionary with backtest results
    """
    # Initialize backtest engine
    engine = BacktestEngine(
        initial_capital=initial_capital,
        commission=commission,
        slippage=slippage
    )
    
    # Generate signals
    signals = strategy.generate_signals(data)
    
    # Run backtest
    results = engine.run_backtest(data, lambda x: signals)
    
    logger.info(f"Completed backtest for {strategy.name}")
    return results


def plot_equity_curve(results: Dict[str, Any], 
                     title: Optional[str] = None,
                     figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot the equity curve from backtest results.
    
    Args:
        results: Dictionary with backtest results
        title: Title for the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    equity_df = results.get('equity_curve')
    if equity_df is None or len(equity_df) == 0:
        logger.warning("No equity data available for plotting")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curve
    ax.plot(equity_df['timestamp'], equity_df['equity'], label='Equity')
    
    # Add initial capital as horizontal line
    initial_capital = results.get('statistics', {}).get('initial_capital', 10000)
    ax.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
    
    # Set title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Equity Curve')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Equity')
    ax.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    return fig


def plot_drawdown(results: Dict[str, Any], 
                 title: Optional[str] = None,
                 figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot the drawdown from backtest results.
    
    Args:
        results: Dictionary with backtest results
        title: Title for the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    equity_df = results.get('equity_curve')
    if equity_df is None or len(equity_df) == 0:
        logger.warning("No equity data available for plotting drawdown")
        return None
    
    # Calculate drawdown
    equity_series = equity_df['equity']
    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot drawdown
    ax.fill_between(equity_df['timestamp'], drawdown, 0, color='red', alpha=0.3)
    ax.plot(equity_df['timestamp'], drawdown, color='red', label='Drawdown')
    
    # Set title and labels
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Drawdown')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.legend()
    
    # Format x-axis dates
    fig.autofmt_xdate()
    
    return fig


def plot_trade_analysis(results: Dict[str, Any], 
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot trade analysis from backtest results.
    
    Args:
        results: Dictionary with backtest results
        title: Title for the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    trades_df = results.get('trades')
    if trades_df is None or len(trades_df) == 0:
        logger.warning("No trade data available for analysis")
        return None
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=figsize)
    
    # Plot trade sizes
    axs[0].bar(range(len(trades_df)), trades_df['quantity'].abs(), 
              color=np.where(trades_df['quantity'] > 0, 'green', 'red'))
    axs[0].set_title('Trade Sizes')
    axs[0].set_xlabel('Trade Number')
    axs[0].set_ylabel('Size')
    
    # Plot cumulative profit/loss if available
    if 'profit' in trades_df.columns:
        cumulative_profit = trades_df['profit'].cumsum()
        axs[1].plot(range(len(trades_df)), cumulative_profit, label='Cumulative P&L')
        axs[1].set_title('Cumulative Profit/Loss')
        axs[1].set_xlabel('Trade Number')
        axs[1].set_ylabel('Profit/Loss')
        axs[1].axhline(y=0, color='r', linestyle='--')
    else:
        axs[1].text(0.5, 0.5, 'Profit data not available', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=axs[1].transAxes)
    
    # Set overall title
    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle('Trade Analysis', fontsize=16)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return fig


def print_statistics(results: Dict[str, Any]):
    """
    Print backtest statistics.
    
    Args:
        results: Dictionary with backtest results
    """
    stats = results.get('statistics', {})
    if not stats:
        logger.warning("No statistics available")
        return
    
    print("\n===== Backtest Statistics =====")
    
    # Performance metrics
    print("\nPerformance Metrics:")
    if 'total_return' in stats:
        print(f"Total Return: {stats['total_return']:.2%}")
    if 'annualized_return' in stats:
        print(f"Annualized Return: {stats['annualized_return']:.2%}")
    if 'sharpe_ratio' in stats:
        print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    if 'max_drawdown' in stats:
        print(f"Maximum Drawdown: {stats['max_drawdown']:.2%}")
    
    # Trade statistics
    print("\nTrade Statistics:")
    if 'total_trades' in stats:
        print(f"Total Trades: {stats['total_trades']}")
    if 'buy_trades' in stats and 'sell_trades' in stats:
        print(f"Buy Trades: {stats['buy_trades']}")
        print(f"Sell Trades: {stats['sell_trades']}")
    if 'win_rate' in stats:
        print(f"Win Rate: {stats['win_rate']:.2%}")
    
    # Final positions
    print("\nFinal State:")
    print(f"Final Capital: ${results.get('final_capital', 0):.2f}")
    
    positions = results.get('final_positions', {})
    if positions:
        print("Final Positions:")
        for symbol, quantity in positions.items():
            print(f"  {symbol}: {quantity}")
    else:
        print("No open positions")
    
    print("\n===============================")


def optimize_strategy_parameters(data: pd.DataFrame,
                               strategy_class,
                               param_grid: Dict[str, List],
                               initial_capital: float = 10000.0,
                               commission: float = 0.0005,
                               slippage: float = 0.0001,
                               metric: str = 'sharpe_ratio') -> Dict[str, Any]:
    """
    Optimize strategy parameters using grid search.
    
    Args:
        data: DataFrame with price data
        strategy_class: Strategy class to optimize
        param_grid: Dictionary of parameter names and values to try
        initial_capital: Initial capital for the backtest
        commission: Commission rate per trade
        slippage: Slippage rate per trade
        metric: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
        
    Returns:
        Dictionary with optimization results
    """
    # Generate all parameter combinations
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    param_combinations = list(itertools.product(*param_values))
    
    # Track best parameters and results
    best_value = -np.inf if metric != 'max_drawdown' else np.inf
    best_params = None
    best_results = None
    all_results = []
    
    # Test each parameter combination
    for params in param_combinations:
        # Create parameter dictionary
        param_dict = dict(zip(param_names, params))
        
        # Create strategy with these parameters
        strategy = strategy_class(**param_dict)
        
        # Run backtest
        results = run_backtest(
            data=data,
            strategy=strategy,
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        
        # Extract metric value
        metric_value = results.get('statistics', {}).get(metric)
        
        # Skip if metric not available
        if metric_value is None:
            continue
        
        # Store results
        result_entry = {
            'params': param_dict,
            metric: metric_value,
            'results': results
        }
        all_results.append(result_entry)
        
        # Update best parameters if better
        is_better = False
        if metric == 'max_drawdown':
            # For drawdown, lower is better
            is_better = metric_value > best_value
        else:
            # For other metrics, higher is better
            is_better = metric_value > best_value
        
        if is_better:
            best_value = metric_value
            best_params = param_dict
            best_results = results
    
    # Sort results
    if metric == 'max_drawdown':
        all_results.sort(key=lambda x: x[metric])
    else:
        all_results.sort(key=lambda x: x[metric], reverse=True)
    
    # Return optimization results
    return {
        'best_params': best_params,
        'best_value': best_value,
        'best_results': best_results,
        'all_results': all_results
    }
