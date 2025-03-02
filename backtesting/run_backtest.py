"""
Script for running backtests.

This script provides a command-line interface for running backtests
with different strategies and parameters.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Union, Tuple, Optional
from datetime import datetime

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import get_settings
from market_data.data_processor import DataProcessor
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
from models.lstm_model.lstm_model import LSTMModel
from models.transformer_model.transformer_model import TransformerModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


def load_data(symbol: str, 
             timeframe: str, 
             start_date: Optional[str] = None, 
             end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load historical data for backtesting.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe for the data
        start_date: Start date for the data
        end_date: End date for the data
        
    Returns:
        DataFrame with historical data
    """
    # Initialize data processor
    data_processor = DataProcessor()
    
    # Fetch OHLCV data
    try:
        # For backtesting, we use synchronous version to simplify
        data = data_processor.fetch_ohlcv_data_sync(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add technical indicators
        data = data_processor.add_technical_indicators(data)
        
        # Handle missing values
        data = data_processor.handle_missing_values(data)
        
        logger.info(f"Loaded {len(data)} data points for {symbol}")
        return data
    
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def load_model(model_type: str, model_path: Optional[str] = None) -> Union[LSTMModel, TransformerModel, None]:
    """
    Load a trained prediction model.
    
    Args:
        model_type: Type of model ('lstm' or 'transformer')
        model_path: Path to the model file
        
    Returns:
        Loaded model instance
    """
    try:
        if model_type.lower() == 'lstm':
            model = LSTMModel()
        elif model_type.lower() == 'transformer':
            model = TransformerModel()
        else:
            logger.error(f"Unknown model type: {model_type}")
            return None
        
        # Load model weights if path provided
        if model_path:
            model.load(model_path)
            logger.info(f"Loaded {model_type} model from {model_path}")
        else:
            logger.warning(f"No model path provided, using untrained {model_type} model")
        
        return model
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None


def create_strategy(strategy_type: str, **kwargs) -> BaseStrategy:
    """
    Create a strategy instance.
    
    Args:
        strategy_type: Type of strategy
        **kwargs: Strategy parameters
        
    Returns:
        Strategy instance
    """
    if strategy_type.lower() == 'trend':
        return TrendFollowingStrategy(**kwargs)
    
    elif strategy_type.lower() == 'mean_reversion':
        return MeanReversionStrategy(**kwargs)
    
    elif strategy_type.lower() == 'predictive':
        # Load model if needed
        model = kwargs.pop('model', None)
        if not model and 'model_type' in kwargs and 'model_path' in kwargs:
            model = load_model(kwargs.pop('model_type'), kwargs.pop('model_path'))
        
        if model:
            return PredictiveModelStrategy(model=model, **kwargs)
        else:
            logger.error("No model provided for predictive strategy")
            return None
    
    else:
        logger.error(f"Unknown strategy type: {strategy_type}")
        return None


def save_results(results: Dict[str, Any], output_dir: str, prefix: str = 'backtest'):
    """
    Save backtest results to files.
    
    Args:
        results: Dictionary with backtest results
        output_dir: Directory to save results
        prefix: Prefix for output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save equity curve
    equity_df = results.get('equity_curve')
    if equity_df is not None and len(equity_df) > 0:
        equity_file = os.path.join(output_dir, f"{prefix}_equity.csv")
        equity_df.to_csv(equity_file, index=False)
        logger.info(f"Saved equity curve to {equity_file}")
    
    # Save trades
    trades_df = results.get('trades')
    if trades_df is not None and len(trades_df) > 0:
        trades_file = os.path.join(output_dir, f"{prefix}_trades.csv")
        trades_df.to_csv(trades_file, index=False)
        logger.info(f"Saved trades to {trades_file}")
    
    # Save statistics
    stats = results.get('statistics')
    if stats:
        import json
        stats_file = os.path.join(output_dir, f"{prefix}_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)
        logger.info(f"Saved statistics to {stats_file}")
    
    # Save plots
    try:
        # Equity curve plot
        fig = plot_equity_curve(results)
        if fig:
            equity_plot_file = os.path.join(output_dir, f"{prefix}_equity.png")
            fig.savefig(equity_plot_file)
            plt.close(fig)
            logger.info(f"Saved equity curve plot to {equity_plot_file}")
        
        # Drawdown plot
        fig = plot_drawdown(results)
        if fig:
            drawdown_plot_file = os.path.join(output_dir, f"{prefix}_drawdown.png")
            fig.savefig(drawdown_plot_file)
            plt.close(fig)
            logger.info(f"Saved drawdown plot to {drawdown_plot_file}")
        
        # Trade analysis plot
        fig = plot_trade_analysis(results)
        if fig:
            trades_plot_file = os.path.join(output_dir, f"{prefix}_trades.png")
            fig.savefig(trades_plot_file)
            plt.close(fig)
            logger.info(f"Saved trade analysis plot to {trades_plot_file}")
    
    except Exception as e:
        logger.error(f"Error saving plots: {e}")


def main():
    """Main function for running backtests."""
    parser = argparse.ArgumentParser(description='Run backtests for trading strategies')
    
    # Data parameters
    parser.add_argument('--symbol', type=str, default='BTC-USD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Data timeframe')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    
    # Strategy parameters
    parser.add_argument('--strategy', type=str, required=True, 
                       choices=['trend', 'mean_reversion', 'predictive'],
                       help='Strategy type')
    
    # Trend following strategy parameters
    parser.add_argument('--fast-period', type=int, default=20, 
                       help='Fast period for trend following strategy')
    parser.add_argument('--slow-period', type=int, default=50, 
                       help='Slow period for trend following strategy')
    
    # Mean reversion strategy parameters
    parser.add_argument('--period', type=int, default=20, 
                       help='Period for mean reversion strategy')
    parser.add_argument('--std-dev', type=float, default=2.0, 
                       help='Standard deviation for mean reversion strategy')
    
    # Predictive model strategy parameters
    parser.add_argument('--model-type', type=str, choices=['lstm', 'transformer'], 
                       help='Model type for predictive strategy')
    parser.add_argument('--model-path', type=str, 
                       help='Path to model file for predictive strategy')
    parser.add_argument('--threshold', type=float, default=0.01, 
                       help='Threshold for predictive strategy')
    
    # Backtest parameters
    parser.add_argument('--initial-capital', type=float, default=10000.0, 
                       help='Initial capital for backtest')
    parser.add_argument('--commission', type=float, default=0.0005, 
                       help='Commission rate per trade')
    parser.add_argument('--slippage', type=float, default=0.0001, 
                       help='Slippage rate per trade')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./backtest_results', 
                       help='Directory for output files')
    parser.add_argument('--prefix', type=str, default='backtest', 
                       help='Prefix for output files')
    
    # Optimization parameters
    parser.add_argument('--optimize', action='store_true', 
                       help='Optimize strategy parameters')
    parser.add_argument('--metric', type=str, default='sharpe_ratio', 
                       choices=['sharpe_ratio', 'total_return', 'max_drawdown'],
                       help='Metric to optimize')
    
    args = parser.parse_args()
    
    # Load data
    data = load_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if len(data) == 0:
        logger.error("No data available for backtesting")
        return
    
    # Run optimization if requested
    if args.optimize:
        logger.info("Running parameter optimization")
        
        # Define parameter grid based on strategy type
        param_grid = {}
        
        if args.strategy == 'trend':
            param_grid = {
                'fast_period': [5, 10, 20, 30],
                'slow_period': [20, 50, 100, 200]
            }
            strategy_class = TrendFollowingStrategy
        
        elif args.strategy == 'mean_reversion':
            param_grid = {
                'period': [10, 20, 30, 50],
                'std_dev': [1.5, 2.0, 2.5, 3.0]
            }
            strategy_class = MeanReversionStrategy
        
        elif args.strategy == 'predictive':
            # Load model for predictive strategy
            model = load_model(args.model_type, args.model_path)
            if not model:
                logger.error("Failed to load model for predictive strategy")
                return
            
            param_grid = {
                'threshold': [0.005, 0.01, 0.015, 0.02]
            }
            
            # Create a wrapper class that includes the model
            class PredictiveStrategyWrapper(PredictiveModelStrategy):
                def __init__(self, threshold):
                    super().__init__(model=model, threshold=threshold)
            
            strategy_class = PredictiveStrategyWrapper
        
        # Run optimization
        opt_results = optimize_strategy_parameters(
            data=data,
            strategy_class=strategy_class,
            param_grid=param_grid,
            initial_capital=args.initial_capital,
            commission=args.commission,
            slippage=args.slippage,
            metric=args.metric
        )
        
        # Print optimization results
        best_params = opt_results.get('best_params')
        best_value = opt_results.get('best_value')
        
        if best_params:
            logger.info(f"Best parameters: {best_params}")
            logger.info(f"Best {args.metric}: {best_value}")
            
            # Save best results
            best_results = opt_results.get('best_results')
            if best_results:
                save_results(
                    results=best_results,
                    output_dir=args.output_dir,
                    prefix=f"{args.prefix}_optimized"
                )
                
                # Print statistics
                print_statistics(best_results)
        else:
            logger.error("Optimization failed to find best parameters")
    
    else:
        # Create strategy with provided parameters
        strategy_params = {}
        
        if args.strategy == 'trend':
            strategy_params = {
                'fast_period': args.fast_period,
                'slow_period': args.slow_period
            }
        
        elif args.strategy == 'mean_reversion':
            strategy_params = {
                'period': args.period,
                'std_dev': args.std_dev
            }
        
        elif args.strategy == 'predictive':
            # Load model for predictive strategy
            model = load_model(args.model_type, args.model_path)
            if not model:
                logger.error("Failed to load model for predictive strategy")
                return
            
            strategy_params = {
                'model': model,
                'threshold': args.threshold
            }
        
        # Create strategy
        strategy = create_strategy(args.strategy, **strategy_params)
        
        if not strategy:
            logger.error("Failed to create strategy")
            return
        
        # Run backtest
        results = run_backtest(
            data=data,
            strategy=strategy,
            initial_capital=args.initial_capital,
            commission=args.commission,
            slippage=args.slippage
        )
        
        # Save results
        save_results(
            results=results,
            output_dir=args.output_dir,
            prefix=args.prefix
        )
        
        # Print statistics
        print_statistics(results)


if __name__ == '__main__':
    main()
