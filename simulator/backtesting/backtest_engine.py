"""
Backtesting engine for evaluating trading strategies and risk management.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

from ..envs.gym_env import TradingEnv, MarketState

@dataclass
class BacktestResults:
    """Container for backtest results and performance metrics."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: pd.DataFrame
    equity_curve: pd.Series
    daily_returns: pd.Series
    risk_metrics: pd.DataFrame

class BacktestEngine:
    def __init__(self,
                 initial_capital: float = 100000.0,
                 risk_free_rate: float = 0.0,
                 trading_fee: float = 0.0002):
        """
        Initialize backtesting engine.
        
        Args:
            initial_capital: Starting capital
            risk_free_rate: Annual risk-free rate for Sharpe ratio calculation
            trading_fee: Trading fee as percentage of trade value
        """
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.trading_fee = trading_fee
        self.logger = logging.getLogger(__name__)
        
        # Initialize environment
        self.env = TradingEnv(
            initial_balance=initial_capital,
            risk_free_rate=risk_free_rate
        )
        
        # Performance tracking
        self.trades = []
        self.equity_curve = []
        self.risk_metrics_history = []

    def run_backtest(self,
                    market_data: pd.DataFrame,
                    strategy,
                    risk_manager) -> BacktestResults:
        """
        Run backtest with given market data and strategy.
        
        Args:
            market_data: Historical market data
            strategy: Trading strategy instance
            risk_manager: Risk management instance
            
        Returns:
            BacktestResults containing performance metrics
        """
        self.logger.info("Starting backtest...")
        
        # Reset environment
        obs = self.env.reset()
        
        # Initialize tracking variables
        current_position = 0.0
        entry_price = 0.0
        trade_count = 0
        
        for idx, row in market_data.iterrows():
            # Update market state
            market_state = MarketState(
                price=row['price'],
                orderbook=row['orderbook'],
                volume=row['volume'],
                funding_rate=row['funding_rate'],
                volatility=row['volatility']
            )
            self.env.update_market_state(market_state)
            
            # Get strategy action
            action = strategy.get_action(obs)
            
            # Apply risk management
            risk_metrics = risk_manager.evaluate_risks(
                self.env.market_state,
                self.env._get_obs()['portfolio']
            )
            action = risk_manager.adjust_action(action, risk_metrics)
            
            # Execute action
            obs, reward, done, info = self.env.step(action)
            
            # Track trade
            if current_position != self.env.position:
                trade = {
                    'timestamp': idx,
                    'type': 'buy' if self.env.position > current_position else 'sell',
                    'price': market_state.price,
                    'size': abs(self.env.position - current_position),
                    'pnl': reward,
                    'portfolio_value': self.env.portfolio_value
                }
                self.trades.append(trade)
                trade_count += 1
                
                current_position = self.env.position
                entry_price = market_state.price
            
            # Track equity and risk metrics
            self.equity_curve.append(self.env.portfolio_value)
            self.risk_metrics_history.append(risk_metrics)
            
            if done:
                break
        
        self.logger.info(f"Backtest completed. Total trades: {trade_count}")
        return self._calculate_performance_metrics()

    def _calculate_performance_metrics(self) -> BacktestResults:
        """Calculate comprehensive performance metrics from backtest results."""
        # Convert tracking data to pandas
        trades_df = pd.DataFrame(self.trades)
        equity_curve = pd.Series(self.equity_curve)
        risk_metrics_df = pd.DataFrame(self.risk_metrics_history)
        
        # Calculate returns
        daily_returns = equity_curve.pct_change().dropna()
        total_return = (equity_curve.iloc[-1] - self.initial_capital) / self.initial_capital
        
        # Risk metrics
        excess_returns = daily_returns - self.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() \
            if excess_returns.std() != 0 else 0
            
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_returns.std() \
            if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
        
        # Maximum drawdown
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Trading metrics
        winning_trades = trades_df[trades_df['pnl'] > 0]
        win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
        
        gross_profits = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
        gross_losses = trades_df[trades_df['pnl'] < 0]['pnl'].sum()
        profit_factor = abs(gross_profits / gross_losses) if gross_losses != 0 else float('inf')
        
        return BacktestResults(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=trades_df,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            risk_metrics=risk_metrics_df
        )

    def generate_report(self, results: BacktestResults, output_dir: Path):
        """Generate comprehensive backtest report with visualizations."""
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save performance metrics
        metrics = {
            'total_return': results.total_return,
            'sharpe_ratio': results.sharpe_ratio,
            'sortino_ratio': results.sortino_ratio,
            'max_drawdown': results.max_drawdown,
            'win_rate': results.win_rate,
            'profit_factor': results.profit_factor
        }
        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Generate plots
        plt.figure(figsize=(12, 8))
        plt.plot(results.equity_curve)
        plt.title('Equity Curve')
        plt.savefig(output_dir / 'equity_curve.png')
        plt.close()
        
        plt.figure(figsize=(12, 8))
        results.daily_returns.hist(bins=50)
        plt.title('Return Distribution')
        plt.savefig(output_dir / 'returns_dist.png')
        plt.close()
        
        # Risk metrics over time
        plt.figure(figsize=(12, 8))
        results.risk_metrics.plot(subplots=True, figsize=(12, 12))
        plt.tight_layout()
        plt.savefig(output_dir / 'risk_metrics.png')
        plt.close()
        
        # Save detailed trade log
        results.trades.to_csv(output_dir / 'trades.csv')
        
        self.logger.info(f"Backtest report generated in {output_dir}")

if __name__ == '__main__':
    # Example usage
    from pathlib import Path
    
    # Initialize components
    engine = BacktestEngine(initial_capital=100000.0)
    
    # Load market data (example)
    market_data = pd.read_csv('market_data.csv')
    
    # Run backtest
    results = engine.run_backtest(
        market_data=market_data,
        strategy=None,  # Add your strategy here
        risk_manager=None  # Add your risk manager here
    )
    
    # Generate report
    output_dir = Path('backtest_results')
    engine.generate_report(results, output_dir)
