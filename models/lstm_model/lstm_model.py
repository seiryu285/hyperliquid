"""
LSTM-based prediction model for time series forecasting.

This module implements an LSTM-based model for predicting price movements
using historical OHLCV data and technical indicators.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple, Optional
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models.base_model import BaseModel
from core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class LSTMModel(BaseModel):
    """
    LSTM-based prediction model for time series forecasting.
    """
    
    def __init__(self, 
                input_dim: int,
                sequence_length: Optional[int] = None,
                prediction_horizon: Optional[int] = None,
                lstm_units: Optional[int] = None,
                dropout: Optional[float] = None,
                recurrent_dropout: Optional[float] = None,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LSTM model.
        
        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences (lookback window)
            prediction_horizon: Number of time steps to predict into the future
            lstm_units: Number of LSTM units
            dropout: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            config: Additional model configuration
        """
        model_name = "lstm_model"
        self.sequence_length = sequence_length or settings.LSTM_SEQUENCE_LENGTH
        prediction_horizon = prediction_horizon or settings.LSTM_PREDICTION_HORIZON
        
        # Initialize base class
        super().__init__(model_name, input_dim, prediction_horizon, config)
        
        # Model specific parameters
        self.lstm_units = lstm_units or settings.LSTM_UNITS
        self.dropout = dropout or settings.LSTM_DROPOUT
        self.recurrent_dropout = recurrent_dropout or settings.LSTM_RECURRENT_DROPOUT
        
        # Build the model
        self.build()
        
        logger.info(f"Initialized LSTM model with units={self.lstm_units}, "
                   f"dropout={self.dropout}, recurrent_dropout={self.recurrent_dropout}")
                   
    def build(self) -> None:
        """Build the LSTM model architecture."""
        model = Sequential([
            LSTM(units=self.lstm_units, 
                 return_sequences=True,
                 dropout=self.dropout,
                 recurrent_dropout=self.recurrent_dropout,
                 input_shape=(self.sequence_length, self.input_dim)),
                 
            LSTM(units=self.lstm_units // 2,
                 return_sequences=False, 
                 dropout=self.dropout,
                 recurrent_dropout=self.recurrent_dropout),
                 
            Dropout(self.dropout),
            
            Dense(units=self.output_dim)
        ])
        
        # Compile the model
        optimizer = Adam(learning_rate=settings.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        logger.info(f"Built LSTM model with architecture: {model.summary()}")
        
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             batch_size: Optional[int] = None,
             epochs: Optional[int] = None,
             early_stopping_patience: Optional[int] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Train the LSTM model on the provided data.
        
        Args:
            X_train: Training features with shape (n_samples, sequence_length, n_features)
            y_train: Training targets with shape (n_samples, prediction_horizon)
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            batch_size: Batch size for training
            epochs: Number of epochs for training
            early_stopping_patience: Patience for early stopping
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history
        """
        if self.model is None:
            logger.error("Model not built yet. Cannot train.")
            return {}
        
        # Set training parameters
        batch_size = batch_size or settings.BATCH_SIZE
        epochs = epochs or settings.EPOCHS
        early_stopping_patience = early_stopping_patience or settings.EARLY_STOPPING_PATIENCE
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=early_stopping_patience // 2,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Add model checkpoint if validation data is provided
        if X_val is not None:
            checkpoint_path = os.path.join(self.model_dir, f"{self.model_name}_best.h5")
            callbacks.append(
                ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True,
                    verbose=1
                )
            )
        
        # Ensure input data has the correct shape
        if len(X_train.shape) != 3:
            logger.error(f"Input data should have shape (n_samples, sequence_length, n_features). "
                        f"Got shape {X_train.shape}")
            return {}
        
        # Train the model
        logger.info(f"Training LSTM model with {X_train.shape[0]} samples over {epochs} epochs")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Store training history
        self.training_history = history.history
        self.is_trained = True
        
        logger.info(f"LSTM model training completed. "
                   f"Final loss: {history.history['loss'][-1]:.4f}, "
                   f"Final val_loss: {history.history['val_loss'][-1]:.4f} "
                   f"if 'val_loss' in history.history else 'N/A'")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the given input data.
        
        Args:
            X: Input features with shape (n_samples, sequence_length, n_features)
            
        Returns:
            Predicted values with shape (n_samples, prediction_horizon)
        """
        if not self.is_trained:
            logger.warning("Model not trained yet. Predictions may be unreliable.")
        
        if self.model is None:
            logger.error("Model not built yet. Cannot predict.")
            return np.array([])
        
        # Ensure input data has the correct shape
        if len(X.shape) != 3:
            logger.error(f"Input data should have shape (n_samples, sequence_length, n_features). "
                        f"Got shape {X.shape}")
            return np.array([])
        
        # Generate predictions
        y_pred = self.model.predict(X)
        
        return y_pred
    
    def _save_model(self, filepath: str) -> None:
        """
        Save the LSTM model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            logger.error("Model not built yet. Cannot save.")
            return
        
        # Save model in HDF5 format
        model_path = f"{filepath}.h5"
        self.model.save(model_path)
        
        # Save training history
        import json
        history_path = f"{filepath}_history.json"
        with open(history_path, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_history = {}
            for key, value in self.training_history.items():
                serializable_history[key] = [float(v) for v in value]
            json.dump(serializable_history, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Training history saved to {history_path}")
    
    def load(self, filepath: str) -> bool:
        """
        Load a trained LSTM model from disk.
        
        Args:
            filepath: Path to the saved model (without extension)
            
        Returns:
            True if loading was successful, False otherwise
        """
        model_path = f"{filepath}.h5"
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return False
        
        try:
            # Load model
            self.model = load_model(model_path)
            self.is_trained = True
            
            # Try to load training history
            import json
            history_path = f"{filepath}_history.json"
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    self.training_history = json.load(f)
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return False
