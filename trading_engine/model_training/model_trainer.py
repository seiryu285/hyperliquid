"""
Model trainer for LSTM and Transformer models.

This module provides functionality for training LSTM and Transformer
models for price prediction.
"""

import os
import logging
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime

from market_data.data_processor import DataProcessor
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class ModelTrainer:
    """
    Class for training LSTM and Transformer models.
    
    This class provides functionality for training LSTM and Transformer
    models for price prediction.
    """
    
    def __init__(self, data_processor: DataProcessor):
        """
        Initialize the model trainer.
        
        Args:
            data_processor: Data processor instance
        """
        self.data_processor = data_processor
        
        # Create models directory if it doesn't exist
        os.makedirs('models/weights', exist_ok=True)
        
        logger.info("Initialized model trainer")
    
    async def train_model(self, 
                        model_config: Dict[str, Any],
                        symbol: str,
                        timeframe: str,
                        start_date: Optional[str] = None,
                        end_date: Optional[str] = None) -> Tuple[Union[LSTMModel, TransformerModel], Dict[str, Any]]:
        """
        Train a model based on configuration.
        
        Args:
            model_config: Model configuration
            symbol: Symbol to train on
            timeframe: Timeframe to train on
            start_date: Start date for training data
            end_date: End date for training data
            
        Returns:
            Tuple of (trained model, training metrics)
        """
        # Get model type
        model_type = model_config.get('type', 'lstm')
        
        # Get training parameters
        lookback = model_config.get('lookback', 20)
        features = model_config.get('features', ['close', 'volume', 'rsi_14', 'macd', 'macd_signal'])
        target = model_config.get('target', 'return_1d')
        test_size = model_config.get('test_size', 0.2)
        epochs = model_config.get('epochs', 100)
        batch_size = model_config.get('batch_size', 32)
        patience = model_config.get('patience', 10)
        
        # Fetch and prepare data
        logger.info(f"Fetching data for {symbol} {timeframe}")
        data = await self.data_processor.fetch_ohlcv_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add technical indicators
        data = self.data_processor.add_technical_indicators(data)
        
        # Add target variable (future returns)
        data = self._add_target_variable(data, target)
        
        # Prepare features and target
        X, y = self._prepare_features_and_target(data, features, target, lookback)
        
        # Split data into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training data shape: {X_train.shape}, {y_train.shape}")
        logger.info(f"Testing data shape: {X_test.shape}, {y_test.shape}")
        
        # Create and train model
        if model_type == 'lstm':
            model = self._create_and_train_lstm(
                X_train, y_train, X_test, y_test,
                model_config, epochs, batch_size, patience
            )
        elif model_type == 'transformer':
            model = self._create_and_train_transformer(
                X_train, y_train, X_test, y_test,
                model_config, epochs, batch_size, patience
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Evaluate model
        metrics = self._evaluate_model(model, X_test, y_test)
        
        # Save model
        model_id = model_config.get('model_id', f"{model_type}_{symbol}_{timeframe}")
        weights_path = f"models/weights/{model_id}.h5"
        model.save_weights(weights_path)
        
        logger.info(f"Saved model weights to {weights_path}")
        logger.info(f"Model evaluation metrics: {metrics}")
        
        return model, metrics
    
    def _add_target_variable(self, data: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Add target variable to data.
        
        Args:
            data: OHLCV data with indicators
            target: Target variable name
            
        Returns:
            Data with target variable
        """
        if target.startswith('return_'):
            # Extract period from target name (e.g., 'return_1d' -> 1)
            period = int(target.split('_')[1].replace('d', ''))
            
            # Calculate future returns
            data[target] = data['close'].pct_change(period).shift(-period)
        
        elif target.startswith('direction_'):
            # Extract period from target name (e.g., 'direction_1d' -> 1)
            period = int(target.split('_')[1].replace('d', ''))
            
            # Calculate future direction (1 for up, 0 for down)
            future_returns = data['close'].pct_change(period).shift(-period)
            data[target] = (future_returns > 0).astype(int)
        
        elif target == 'volatility':
            # Calculate future volatility (standard deviation of returns)
            data[target] = data['close'].pct_change().rolling(window=5).std().shift(-1)
        
        else:
            logger.warning(f"Unknown target variable: {target}")
        
        return data
    
    def _prepare_features_and_target(self, 
                                   data: pd.DataFrame, 
                                   features: List[str],
                                   target: str,
                                   lookback: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for model training.
        
        Args:
            data: OHLCV data with indicators and target
            features: List of feature columns
            target: Target column
            lookback: Number of lookback periods
            
        Returns:
            Tuple of (features, target) as numpy arrays
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
        
        # Get target data
        target_data = data[target].fillna(0).values
        
        # Create sequences
        X, y = [], []
        for i in range(len(feature_data) - lookback):
            X.append(feature_data.values[i:i+lookback])
            y.append(target_data[i+lookback])
        
        return np.array(X), np.array(y)
    
    def _create_and_train_lstm(self,
                             X_train: np.ndarray,
                             y_train: np.ndarray,
                             X_test: np.ndarray,
                             y_test: np.ndarray,
                             model_config: Dict[str, Any],
                             epochs: int,
                             batch_size: int,
                             patience: int) -> LSTMModel:
        """
        Create and train LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target
            model_config: Model configuration
            epochs: Number of epochs
            batch_size: Batch size
            patience: Patience for early stopping
            
        Returns:
            Trained LSTM model
        """
        # Get model parameters
        input_shape = X_train.shape[1:]
        units = model_config.get('units', 64)
        dropout = model_config.get('dropout', 0.2)
        
        # Create model
        model = LSTMModel(
            input_shape=input_shape,
            units=units,
            dropout=dropout
        )
        
        # Train model
        model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        return model
    
    def _create_and_train_transformer(self,
                                    X_train: np.ndarray,
                                    y_train: np.ndarray,
                                    X_test: np.ndarray,
                                    y_test: np.ndarray,
                                    model_config: Dict[str, Any],
                                    epochs: int,
                                    batch_size: int,
                                    patience: int) -> TransformerModel:
        """
        Create and train Transformer model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Testing features
            y_test: Testing target
            model_config: Model configuration
            epochs: Number of epochs
            batch_size: Batch size
            patience: Patience for early stopping
            
        Returns:
            Trained Transformer model
        """
        # Get model parameters
        input_shape = X_train.shape[1:]
        head_size = model_config.get('head_size', 128)
        num_heads = model_config.get('num_heads', 2)
        ff_dim = model_config.get('ff_dim', 128)
        dropout = model_config.get('dropout', 0.2)
        
        # Create model
        model = TransformerModel(
            input_shape=input_shape,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        # Train model
        model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_test,
            y_val=y_test,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience
        )
        
        return model
    
    def _evaluate_model(self, 
                      model: Union[LSTMModel, TransformerModel],
                      X_test: np.ndarray,
                      y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model
            X_test: Testing features
            y_test: Testing target
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = np.mean((y_pred - y_test) ** 2)
        mae = np.mean(np.abs(y_pred - y_test))
        
        # Calculate directional accuracy
        direction_pred = np.sign(y_pred)
        direction_true = np.sign(y_test)
        directional_accuracy = np.mean(direction_pred == direction_true)
        
        # Return metrics
        return {
            'mse': float(mse),
            'mae': float(mae),
            'directional_accuracy': float(directional_accuracy)
        }
