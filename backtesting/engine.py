"""
Backtesting engine for trading strategies.

This module provides a framework for backtesting trading strategies
using historical price data.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple, Optional, Callable
from datetime import datetime

from core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class BacktestEngine:
    """
    Base class for backtesting trading strategies.
    """
    
    def __init__(self, 
                initial_capital: float = 10000.0,
                commission: float = 0.0005,  # 0.05% commission
                slippage: float = 0.0001,    # 0.01% slippage
                ):
        """
        Initialize the backtest engine.
        
        Args:
            initial_capital: Initial capital for the backtest
            commission: Commission rate per trade (as a fraction)
            slippage: Slippage rate per trade (as a fraction)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Backtest state
        self.capital = initial_capital
        self.positions = {}  # symbol -> quantity
        self.trades = []
        self.equity_curve = []
        
        logger.info(f"Initialized backtest engine with {initial_capital} capital")
    
    def reset(self):
        """Reset the backtest state."""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
        logger.info("Reset backtest engine")
    
    def calculate_order_cost(self, price: float, quantity: float) -> float:
        """
        Calculate the total cost of an order including commission.
        
        Args:
            price: Price per unit
            quantity: Quantity to buy/sell
            
        Returns:
            Total cost including commission
        """
        base_cost = price * quantity
        commission_cost = base_cost * self.commission
        return base_cost + commission_cost
    
    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """
        Apply slippage to the price.
        
        Args:
            price: Original price
            is_buy: Whether this is a buy order
            
        Returns:
            Price with slippage applied
        """
        if is_buy:
            # For buy orders, price increases
            return price * (1 + self.slippage)
        else:
            # For sell orders, price decreases
            return price * (1 - self.slippage)
    
    def execute_order(self, 
                     symbol: str, 
                     price: float, 
                     quantity: float, 
                     order_type: str,
                     timestamp: datetime) -> Dict[str, Any]:
        """
        Execute a simulated order.
        
        Args:
            symbol: Trading symbol
            price: Price per unit
            quantity: Quantity to buy/sell (positive for buy, negative for sell)
            order_type: Type of order ('market', 'limit', etc.)
            timestamp: Time of the order
            
        Returns:
            Dictionary with order execution details
        """
        is_buy = quantity > 0
        
        # Apply slippage to the price
        execution_price = self.apply_slippage(price, is_buy)
        
        # Calculate order cost
        order_cost = self.calculate_order_cost(execution_price, abs(quantity))
        
        # Check if we have enough capital for buy orders
        if is_buy and order_cost > self.capital:
            logger.warning(f"Insufficient capital for order: {order_cost} > {self.capital}")
            return {
                'success': False,
                'error': 'Insufficient capital',
                'symbol': symbol,
                'price': execution_price,
                'quantity': quantity,
                'timestamp': timestamp
            }
        
        # Update capital
        if is_buy:
            self.capital -= order_cost
        else:
            self.capital += order_cost
        
        # Update positions
        current_position = self.positions.get(symbol, 0)
        new_position = current_position + quantity
        
        if abs(new_position) < 1e-10:  # Close to zero
            if symbol in self.positions:
                del self.positions[symbol]
        else:
            self.positions[symbol] = new_position
        
        # Record the trade
        trade = {
            'symbol': symbol,
            'price': execution_price,
            'quantity': quantity,
            'cost': order_cost,
            'type': order_type,
            'timestamp': timestamp,
            'capital_after': self.capital
        }
        self.trades.append(trade)
        
        logger.info(f"Executed {order_type} order: {symbol} {quantity} @ {execution_price}")
        return {
            'success': True,
            'trade': trade
        }
    
    def update_equity(self, prices: Dict[str, float], timestamp: datetime):
        """
        Update the equity curve with current positions and prices.
        
        Args:
            prices: Dictionary mapping symbols to current prices
            timestamp: Current timestamp
        """
        # Calculate position values
        position_value = 0
        for symbol, quantity in self.positions.items():
            if symbol in prices:
                position_value += prices[symbol] * quantity
        
        # Calculate total equity
        equity = self.capital + position_value
        
        # Record equity
        self.equity_curve.append({
            'timestamp': timestamp,
            'capital': self.capital,
            'position_value': position_value,
            'equity': equity
        })
    
    def get_returns(self) -> pd.DataFrame:
        """
        Calculate returns from the equity curve.
        
        Returns:
            DataFrame with returns data
        """
        if not self.equity_curve:
            logger.warning("No equity data available")
            return pd.DataFrame()
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate returns
        equity_df['return'] = equity_df['equity'].pct_change()
        equity_df['cumulative_return'] = (1 + equity_df['return']).cumprod() - 1
        
        return equity_df
    
    def get_trade_history(self) -> pd.DataFrame:
        """
        Get the trade history as a DataFrame.
        
        Returns:
            DataFrame with trade history
        """
        if not self.trades:
            logger.warning("No trades available")
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate performance statistics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self.equity_curve:
            logger.warning("No equity data available for statistics")
            return {}
        
        equity_df = self.get_returns()
        trades_df = self.get_trade_history()
        
        # Basic statistics
        stats = {}
        
        # Return statistics
        if len(equity_df) > 1:
            total_return = equity_df['equity'].iloc[-1] / self.initial_capital - 1
            stats['total_return'] = total_return
            stats['annualized_return'] = (1 + total_return) ** (252 / len(equity_df)) - 1
            
            # Risk statistics
            returns = equity_df['return'].dropna()
            if len(returns) > 1:
                stats['volatility'] = returns.std() * np.sqrt(252)
                stats['sharpe_ratio'] = (stats['annualized_return'] / stats['volatility'] 
                                        if stats['volatility'] > 0 else 0)
                
                # Drawdown
                equity_series = equity_df['equity']
                rolling_max = equity_series.cummax()
                drawdown = (equity_series - rolling_max) / rolling_max
                stats['max_drawdown'] = drawdown.min()
        
        # Trade statistics
        if len(trades_df) > 0:
            stats['total_trades'] = len(trades_df)
            
            # Separate buy and sell trades
            buy_trades = trades_df[trades_df['quantity'] > 0]
            sell_trades = trades_df[trades_df['quantity'] < 0]
            
            stats['buy_trades'] = len(buy_trades)
            stats['sell_trades'] = len(sell_trades)
            
            # Calculate win rate (if possible)
            if 'profit' in trades_df.columns:
                winning_trades = trades_df[trades_df['profit'] > 0]
                stats['win_rate'] = len(winning_trades) / len(trades_df)
        
        return stats
    
    def run_backtest(self, 
                    data: pd.DataFrame, 
                    strategy_func: Callable,
                    **strategy_params) -> Dict[str, Any]:
        """
        Run a backtest using the provided data and strategy.
        
        Args:
            data: DataFrame with price data
            strategy_func: Function that generates trading signals
            **strategy_params: Parameters for the strategy function
            
        Returns:
            Dictionary with backtest results
        """
        # Reset the backtest state
        self.reset()
        
        # Ensure data is sorted by date
        if 'date' in data.columns:
            data = data.sort_values('date')
        
        # Run the strategy function to get signals
        signals = strategy_func(data, **strategy_params)
        
        # Process signals and execute orders
        for i, row in signals.iterrows():
            timestamp = row.get('date', pd.Timestamp(f'2000-01-01 {i:04d}'))
            symbol = row.get('symbol', 'UNKNOWN')
            
            # Check for trade signals
            if 'signal' in row and row['signal'] != 0:
                price = row.get('close', row.get('price', 0))
                quantity = row['signal']
                
                self.execute_order(
                    symbol=symbol,
                    price=price,
                    quantity=quantity,
                    order_type='market',
                    timestamp=timestamp
                )
            
            # Update equity curve
            prices = {symbol: row.get('close', row.get('price', 0))}
            self.update_equity(prices, timestamp)
        
        # Calculate statistics
        stats = self.calculate_statistics()
        
        # Prepare results
        results = {
            'statistics': stats,
            'equity_curve': self.get_returns(),
            'trades': self.get_trade_history(),
            'final_capital': self.capital,
            'final_positions': self.positions.copy()
        }
        
        logger.info(f"Backtest completed with final capital: {self.capital}")
        return results
