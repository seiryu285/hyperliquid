"""
Trading strategy base classes and implementations.

This module provides base classes for implementing trading strategies,
as well as some common strategy implementations.
"""

import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Tuple, Optional

# Configure logging
logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Abstract base class for trading strategies.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize the strategy.
        
        Args:
            name: Name of the strategy
        """
        self.name = name
        logger.info(f"Initialized {name} strategy")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from the input data.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            DataFrame with added signal column (positive for buy, negative for sell)
        """
        pass
    
    def calculate_position_sizes(self, 
                               signals: pd.DataFrame, 
                               capital: float,
                               risk_per_trade: float = 0.02) -> pd.DataFrame:
        """
        Calculate position sizes based on signals and available capital.
        
        Args:
            signals: DataFrame with trading signals
            capital: Available capital
            risk_per_trade: Fraction of capital to risk per trade
            
        Returns:
            DataFrame with position sizes
        """
        # Make a copy to avoid modifying the original
        result = signals.copy()
        
        # Calculate position sizes
        if 'signal' in result.columns:
            # Initialize position size column
            result['position_size'] = 0.0
            
            # For each signal, calculate position size
            for i, row in result.iterrows():
                if row['signal'] != 0:
                    # Determine direction (buy/sell)
                    direction = np.sign(row['signal'])
                    
                    # Calculate position size based on risk
                    risk_amount = capital * risk_per_trade
                    price = row.get('close', row.get('price', 0))
                    
                    if price > 0:
                        # Simple position sizing based on fixed risk amount
                        position_size = direction * (risk_amount / price)
                        result.at[i, 'position_size'] = position_size
        
        return result


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy based on moving averages.
    """
    
    def __init__(self, 
                fast_period: int = 20, 
                slow_period: int = 50,
                name: str = "TrendFollowing"):
        """
        Initialize the trend following strategy.
        
        Args:
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            name: Name of the strategy
        """
        super().__init__(name)
        self.fast_period = fast_period
        self.slow_period = slow_period
        
        logger.info(f"Initialized trend following strategy with fast_period={fast_period}, "
                   f"slow_period={slow_period}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on moving average crossovers.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added signal column
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate moving averages if not already present
        if f'sma_{self.fast_period}' not in df.columns:
            df[f'sma_{self.fast_period}'] = df['close'].rolling(window=self.fast_period).mean()
        
        if f'sma_{self.slow_period}' not in df.columns:
            df[f'sma_{self.slow_period}'] = df['close'].rolling(window=self.slow_period).mean()
        
        # Initialize signal column
        df['signal'] = 0
        
        # Calculate crossover signals
        fast_ma = df[f'sma_{self.fast_period}']
        slow_ma = df[f'sma_{self.slow_period}']
        
        # Buy signal: fast MA crosses above slow MA
        buy_condition = (fast_ma.shift(1) <= slow_ma.shift(1)) & (fast_ma > slow_ma)
        df.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: fast MA crosses below slow MA
        sell_condition = (fast_ma.shift(1) >= slow_ma.shift(1)) & (fast_ma < slow_ma)
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy based on Bollinger Bands.
    """
    
    def __init__(self, 
                period: int = 20, 
                std_dev: float = 2.0,
                name: str = "MeanReversion"):
        """
        Initialize the mean reversion strategy.
        
        Args:
            period: Period for the moving average
            std_dev: Number of standard deviations for the bands
            name: Name of the strategy
        """
        super().__init__(name)
        self.period = period
        self.std_dev = std_dev
        
        logger.info(f"Initialized mean reversion strategy with period={period}, "
                   f"std_dev={std_dev}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            DataFrame with added signal column
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Calculate Bollinger Bands if not already present
        if 'bb_middle' not in df.columns:
            df['bb_middle'] = df['close'].rolling(window=self.period).mean()
            rolling_std = df['close'].rolling(window=self.period).std()
            df['bb_upper'] = df['bb_middle'] + (rolling_std * self.std_dev)
            df['bb_lower'] = df['bb_middle'] - (rolling_std * self.std_dev)
        
        # Initialize signal column
        df['signal'] = 0
        
        # Buy signal: price crosses below lower band
        buy_condition = (df['close'].shift(1) <= df['bb_lower'].shift(1)) & (df['close'] > df['bb_lower'])
        df.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: price crosses above upper band
        sell_condition = (df['close'].shift(1) >= df['bb_upper'].shift(1)) & (df['close'] < df['bb_upper'])
        df.loc[sell_condition, 'signal'] = -1
        
        return df


class PredictiveModelStrategy(BaseStrategy):
    """
    Strategy based on predictions from a machine learning model.
    """
    
    def __init__(self, 
                model,
                threshold: float = 0.01,
                name: str = "PredictiveModel"):
        """
        Initialize the predictive model strategy.
        
        Args:
            model: Trained prediction model
            threshold: Threshold for generating signals (as a fraction)
            name: Name of the strategy
        """
        super().__init__(name)
        self.model = model
        self.threshold = threshold
        
        logger.info(f"Initialized predictive model strategy with threshold={threshold}")
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on model predictions.
        
        Args:
            data: DataFrame with price and feature data
            
        Returns:
            DataFrame with added signal column
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Check if predictions are already in the data
        if 'prediction' not in df.columns:
            logger.warning("No prediction column found in data")
            df['signal'] = 0
            return df
        
        # Initialize signal column
        df['signal'] = 0
        
        # Calculate predicted returns
        df['predicted_return'] = df['prediction'] / df['close'] - 1
        
        # Buy signal: predicted return exceeds threshold
        buy_condition = df['predicted_return'] > self.threshold
        df.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: predicted return below negative threshold
        sell_condition = df['predicted_return'] < -self.threshold
        df.loc[sell_condition, 'signal'] = -1
        
        return df
