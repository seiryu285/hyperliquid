"""
Predictive trading engine for executing trading strategies based on ML models.

This module provides the PredictiveEngine class for executing
trading strategies using machine learning models.
"""

import logging
import asyncio
import time
import numpy as np
import pandas as pd
import os
from typing import Dict, List, Optional, Any, Callable, Coroutine, Union, Tuple
from datetime import datetime

from market_data.data_processor import DataProcessor
from order_management.order_manager import OrderManager
from order_management.execution_engine import ExecutionEngine
from order_management.order_types import (
    OrderSide, OrderType, OrderStatus, OrderTimeInForce,
    Order, OrderFill, Position, OrderRequest, OrderCancelRequest
)
from risk_management.risk_monitoring.risk_monitor import RiskMonitor
from risk_management.position_sizing.kelly_criterion import KellyCriterion
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from .engine import TradingEngine
from core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class PredictiveEngine(TradingEngine):
    """
    Trading engine for executing strategies based on predictive models.
    
    This class extends the base TradingEngine to use machine learning
    models for generating trading signals.
    """
    
    def __init__(self, 
                order_manager: OrderManager,
                execution_engine: ExecutionEngine,
                data_processor: DataProcessor,
                risk_monitor: Optional[RiskMonitor] = None,
                kelly_criterion: Optional[KellyCriterion] = None,
                dry_run: bool = True,
                models_dir: str = "models/weights"):
        """
        Initialize the predictive trading engine.
        
        Args:
            order_manager: Order manager instance
            execution_engine: Execution engine instance
            data_processor: Data processor instance
            risk_monitor: Risk monitor instance
            kelly_criterion: Kelly criterion instance
            dry_run: Whether to run in dry run mode
            models_dir: Directory containing model weights
        """
        super().__init__(
            order_manager=order_manager,
            execution_engine=execution_engine,
            data_processor=data_processor,
            risk_monitor=risk_monitor,
            kelly_criterion=kelly_criterion,
            dry_run=dry_run
        )
        
        # Model cache
        self.models = {}
        self.models_dir = models_dir
        
        # Create models directory if it doesn't exist
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.info("Initialized predictive trading engine")
    
    async def _generate_strategy_signals(self, 
                                       strategy_type: str,
                                       symbol: str,
                                       timeframe: str,
                                       params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate signals for a specific strategy, symbol, and timeframe.
        
        Args:
            strategy_type: Type of strategy
            symbol: Symbol
            timeframe: Timeframe
            params: Strategy parameters
            
        Returns:
            List of signals
        """
        signals = []
        
        try:
            # Get data
            data = await self.data_processor.fetch_ohlcv_data(
                symbol=symbol,
                timeframe=timeframe
            )
            
            # Add technical indicators
            data = self.data_processor.add_technical_indicators(data)
            
            # Check if we have enough data
            if len(data) < 50:
                logger.warning(f"Not enough data for {symbol} {timeframe}")
                return []
            
            # Generate signals based on strategy type
            if strategy_type == 'lstm':
                signals = await self._generate_lstm_signals(data, symbol, timeframe, params)
            elif strategy_type == 'transformer':
                signals = await self._generate_transformer_signals(data, symbol, timeframe, params)
            elif strategy_type == 'trend_following':
                signals = self._generate_trend_following_signals(data, symbol, timeframe, params)
            elif strategy_type == 'mean_reversion':
                signals = self._generate_mean_reversion_signals(data, symbol, timeframe, params)
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}")
        
        except Exception as e:
            logger.error(f"Error generating signals for {symbol} {timeframe}: {e}")
        
        return signals
    
    async def _generate_lstm_signals(self, 
                                   data: pd.DataFrame,
                                   symbol: str,
                                   timeframe: str,
                                   params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate signals using LSTM model.
        
        Args:
            data: OHLCV data with indicators
            symbol: Symbol
            timeframe: Timeframe
            params: Strategy parameters
            
        Returns:
            List of signals
        """
        signals = []
        
        try:
            # Get model parameters
            model_id = params.get('model_id', f'lstm_{symbol}_{timeframe}')
            lookback = params.get('lookback', 20)
            features = params.get('features', ['close', 'volume', 'rsi_14', 'macd', 'macd_signal'])
            target = params.get('target', 'return_1d')
            threshold = params.get('threshold', 0.01)
            position_size = params.get('position_size', 1.0)
            
            # Load or create model
            model = self._get_or_create_model(
                model_id=model_id,
                model_type='lstm',
                params=params
            )
            
            # Prepare data
            X = self._prepare_features(data, features, lookback)
            
            # Make prediction
            prediction = model.predict(X)
            
            # Get latest prediction
            latest_pred = prediction[-1][0]
            
            # Generate signal if prediction exceeds threshold
            if abs(latest_pred) > threshold:
                side = 'buy' if latest_pred > 0 else 'sell'
                
                # Calculate win rate and risk/reward for Kelly sizing
                win_rate = 0.5 + abs(latest_pred) * 0.5  # Simple heuristic
                risk_reward = 1.0  # Default risk/reward
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': position_size,
                    'price': data['close'].iloc[-1],
                    'order_type': 'market',
                    'strategy': 'lstm',
                    'timeframe': timeframe,
                    'prediction': float(latest_pred),
                    'win_rate': win_rate,
                    'risk_reward': risk_reward,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                signals.append(signal)
                logger.info(f"Generated LSTM signal for {symbol}: {side} (pred={latest_pred:.4f})")
        
        except Exception as e:
            logger.error(f"Error generating LSTM signals for {symbol} {timeframe}: {e}")
        
        return signals
    
    async def _generate_transformer_signals(self, 
                                          data: pd.DataFrame,
                                          symbol: str,
                                          timeframe: str,
                                          params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate signals using Transformer model.
        
        Args:
            data: OHLCV data with indicators
            symbol: Symbol
            timeframe: Timeframe
            params: Strategy parameters
            
        Returns:
            List of signals
        """
        signals = []
        
        try:
            # Get model parameters
            model_id = params.get('model_id', f'transformer_{symbol}_{timeframe}')
            lookback = params.get('lookback', 20)
            features = params.get('features', ['close', 'volume', 'rsi_14', 'macd', 'macd_signal'])
            target = params.get('target', 'return_1d')
            threshold = params.get('threshold', 0.01)
            position_size = params.get('position_size', 1.0)
            
            # Load or create model
            model = self._get_or_create_model(
                model_id=model_id,
                model_type='transformer',
                params=params
            )
            
            # Prepare data
            X = self._prepare_features(data, features, lookback)
            
            # Make prediction
            prediction = model.predict(X)
            
            # Get latest prediction
            latest_pred = prediction[-1][0]
            
            # Generate signal if prediction exceeds threshold
            if abs(latest_pred) > threshold:
                side = 'buy' if latest_pred > 0 else 'sell'
                
                # Calculate win rate and risk/reward for Kelly sizing
                win_rate = 0.5 + abs(latest_pred) * 0.5  # Simple heuristic
                risk_reward = 1.0  # Default risk/reward
                
                # Create signal
                signal = {
                    'symbol': symbol,
                    'side': side,
                    'quantity': position_size,
                    'price': data['close'].iloc[-1],
                    'order_type': 'market',
                    'strategy': 'transformer',
                    'timeframe': timeframe,
                    'prediction': float(latest_pred),
                    'win_rate': win_rate,
                    'risk_reward': risk_reward,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                signals.append(signal)
                logger.info(f"Generated Transformer signal for {symbol}: {side} (pred={latest_pred:.4f})")
        
        except Exception as e:
            logger.error(f"Error generating Transformer signals for {symbol} {timeframe}: {e}")
        
        return signals
    
    def _generate_trend_following_signals(self, 
                                        data: pd.DataFrame,
                                        symbol: str,
                                        timeframe: str,
                                        params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate signals using trend following strategy.
        
        Args:
            data: OHLCV data with indicators
            symbol: Symbol
            timeframe: Timeframe
            params: Strategy parameters
            
        Returns:
            List of signals
        """
        signals = []
        
        try:
            # Get strategy parameters
            fast_ma = params.get('fast_ma', 10)
            slow_ma = params.get('slow_ma', 50)
            position_size = params.get('position_size', 1.0)
            
            # Calculate moving averages if not already in data
            fast_col = f'ma_{fast_ma}'
            slow_col = f'ma_{slow_ma}'
            
            if fast_col not in data.columns:
                data[fast_col] = data['close'].rolling(window=fast_ma).mean()
            
            if slow_col not in data.columns:
                data[slow_col] = data['close'].rolling(window=slow_ma).mean()
            
            # Get current and previous values
            curr_fast = data[fast_col].iloc[-1]
            prev_fast = data[fast_col].iloc[-2]
            curr_slow = data[slow_col].iloc[-1]
            prev_slow = data[slow_col].iloc[-2]
            
            # Check for crossover
            prev_diff = prev_fast - prev_slow
            curr_diff = curr_fast - curr_slow
            
            # Generate signal on crossover
            if prev_diff <= 0 and curr_diff > 0:
                # Bullish crossover
                signal = {
                    'symbol': symbol,
                    'side': 'buy',
                    'quantity': position_size,
                    'price': data['close'].iloc[-1],
                    'order_type': 'market',
                    'strategy': 'trend_following',
                    'timeframe': timeframe,
                    'fast_ma': curr_fast,
                    'slow_ma': curr_slow,
                    'win_rate': 0.55,  # Historical win rate for this strategy
                    'risk_reward': 1.2,  # Historical risk/reward for this strategy
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                signals.append(signal)
                logger.info(f"Generated trend following BUY signal for {symbol}")
            
            elif prev_diff >= 0 and curr_diff < 0:
                # Bearish crossover
                signal = {
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': position_size,
                    'price': data['close'].iloc[-1],
                    'order_type': 'market',
                    'strategy': 'trend_following',
                    'timeframe': timeframe,
                    'fast_ma': curr_fast,
                    'slow_ma': curr_slow,
                    'win_rate': 0.55,  # Historical win rate for this strategy
                    'risk_reward': 1.2,  # Historical risk/reward for this strategy
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                signals.append(signal)
                logger.info(f"Generated trend following SELL signal for {symbol}")
        
        except Exception as e:
            logger.error(f"Error generating trend following signals for {symbol} {timeframe}: {e}")
        
        return signals
    
    def _generate_mean_reversion_signals(self, 
                                       data: pd.DataFrame,
                                       symbol: str,
                                       timeframe: str,
                                       params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate signals using mean reversion strategy.
        
        Args:
            data: OHLCV data with indicators
            symbol: Symbol
            timeframe: Timeframe
            params: Strategy parameters
            
        Returns:
            List of signals
        """
        signals = []
        
        try:
            # Get strategy parameters
            rsi_period = params.get('rsi_period', 14)
            rsi_oversold = params.get('rsi_oversold', 30)
            rsi_overbought = params.get('rsi_overbought', 70)
            position_size = params.get('position_size', 1.0)
            
            # Calculate RSI if not already in data
            rsi_col = f'rsi_{rsi_period}'
            
            if rsi_col not in data.columns:
                delta = data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.rolling(window=rsi_period).mean()
                avg_loss = loss.rolling(window=rsi_period).mean()
                rs = avg_gain / avg_loss
                data[rsi_col] = 100 - (100 / (1 + rs))
            
            # Get current and previous RSI
            curr_rsi = data[rsi_col].iloc[-1]
            prev_rsi = data[rsi_col].iloc[-2]
            
            # Generate signals based on RSI
            if prev_rsi <= rsi_oversold and curr_rsi > rsi_oversold:
                # RSI crossing up from oversold
                signal = {
                    'symbol': symbol,
                    'side': 'buy',
                    'quantity': position_size,
                    'price': data['close'].iloc[-1],
                    'order_type': 'market',
                    'strategy': 'mean_reversion',
                    'timeframe': timeframe,
                    'rsi': curr_rsi,
                    'win_rate': 0.6,  # Historical win rate for this strategy
                    'risk_reward': 1.0,  # Historical risk/reward for this strategy
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                signals.append(signal)
                logger.info(f"Generated mean reversion BUY signal for {symbol} (RSI={curr_rsi:.2f})")
            
            elif prev_rsi >= rsi_overbought and curr_rsi < rsi_overbought:
                # RSI crossing down from overbought
                signal = {
                    'symbol': symbol,
                    'side': 'sell',
                    'quantity': position_size,
                    'price': data['close'].iloc[-1],
                    'order_type': 'market',
                    'strategy': 'mean_reversion',
                    'timeframe': timeframe,
                    'rsi': curr_rsi,
                    'win_rate': 0.6,  # Historical win rate for this strategy
                    'risk_reward': 1.0,  # Historical risk/reward for this strategy
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                signals.append(signal)
                logger.info(f"Generated mean reversion SELL signal for {symbol} (RSI={curr_rsi:.2f})")
        
        except Exception as e:
            logger.error(f"Error generating mean reversion signals for {symbol} {timeframe}: {e}")
        
        return signals
    
    def _get_or_create_model(self, model_id: str, model_type: str, params: Dict[str, Any]) -> Any:
        """
        Get or create a model.
        
        Args:
            model_id: Model ID
            model_type: Model type ('lstm' or 'transformer')
            params: Model parameters
            
        Returns:
            Model instance
        """
        # Check if model exists in cache
        if model_id in self.models:
            return self.models[model_id]
        
        # Default weights path in models directory
        default_weights_path = os.path.join(self.models_dir, f"{model_id}.h5")
        
        # Create new model
        if model_type == 'lstm':
            # Get LSTM parameters
            input_shape = params.get('input_shape', (20, 5))
            units = params.get('units', 64)
            dropout = params.get('dropout', 0.2)
            
            # Create LSTM model
            model = LSTMModel(
                input_shape=input_shape,
                units=units,
                dropout=dropout
            )
            
            # Load weights if available
            weights_path = params.get('weights_path', default_weights_path)
            if os.path.exists(weights_path):
                try:
                    model.load_weights(weights_path)
                    logger.info(f"Loaded LSTM weights from {weights_path}")
                except Exception as e:
                    logger.warning(f"Failed to load LSTM weights from {weights_path}: {e}")
            else:
                logger.warning(f"LSTM weights file not found at {weights_path}. Using untrained model.")
        
        elif model_type == 'transformer':
            # Get Transformer parameters
            input_shape = params.get('input_shape', (20, 5))
            head_size = params.get('head_size', 128)
            num_heads = params.get('num_heads', 2)
            ff_dim = params.get('ff_dim', 128)
            dropout = params.get('dropout', 0.2)
            
            # Create Transformer model
            model = TransformerModel(
                input_shape=input_shape,
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout
            )
            
            # Load weights if available
            weights_path = params.get('weights_path', default_weights_path)
            if os.path.exists(weights_path):
                try:
                    model.load_weights(weights_path)
                    logger.info(f"Loaded Transformer weights from {weights_path}")
                except Exception as e:
                    logger.warning(f"Failed to load Transformer weights from {weights_path}: {e}")
            else:
                logger.warning(f"Transformer weights file not found at {weights_path}. Using untrained model.")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Cache model
        self.models[model_id] = model
        
        return model
    
    def _prepare_features(self, data: pd.DataFrame, features: List[str], lookback: int) -> np.ndarray:
        """
        Prepare features for model input.
        
        Args:
            data: OHLCV data with indicators
            features: List of feature columns
            lookback: Number of lookback periods
            
        Returns:
            Numpy array of features
        """
        # Select features
        feature_data = data[features].copy()
        
        # Handle missing values
        feature_data = feature_data.fillna(method='ffill').fillna(0)
        
        # Normalize features
        for col in feature_data.columns:
            mean = feature_data[col].mean()
            std = feature_data[col].std()
            if std > 0:
                feature_data[col] = (feature_data[col] - mean) / std
        
        # Convert to numpy array
        feature_array = feature_data.values
        
        # Create sequences
        X = []
        for i in range(len(feature_array) - lookback):
            X.append(feature_array[i:i+lookback])
        
        return np.array(X)
