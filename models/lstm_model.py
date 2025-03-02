"""
LSTM model for price prediction.

This module provides an LSTM model for price prediction.
"""

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, List, Optional, Any, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)


class LSTMModel:
    """
    LSTM model for price prediction.
    
    This class provides an LSTM model for price prediction.
    """
    
    def __init__(self, 
               input_shape: Tuple[int, int],
               units: int = 64,
               dropout: float = 0.2):
        """
        Initialize the LSTM model.
        
        Args:
            input_shape: Shape of input data (sequence_length, num_features)
            units: Number of LSTM units
            dropout: Dropout rate
        """
        self.input_shape = input_shape
        self.units = units
        self.dropout = dropout
        
        # Build model
        self.model = self._build_model()
        
        logger.info(f"Initialized LSTM model with input shape {input_shape}")
    
    def _build_model(self) -> Model:
        """
        Build the LSTM model.
        
        Returns:
            Keras model
        """
        # Input layer
        inputs = Input(shape=self.input_shape)
        
        # LSTM layers
        x = LSTM(units=self.units, return_sequences=True)(inputs)
        x = Dropout(self.dropout)(x)
        x = LSTM(units=self.units)(x)
        x = Dropout(self.dropout)(x)
        
        # Output layer
        outputs = Dense(units=1, activation='linear')(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train(self, 
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            epochs: int = 100,
            batch_size: int = 32,
            patience: int = 10):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of epochs
            batch_size: Batch size
            patience: Patience for early stopping
        """
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=patience,
                restore_best_weights=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-6
            )
        ]
        
        # Train model
        history = self.model.fit(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Log training results
        val_loss = history.history['val_loss'][-1] if X_val is not None else None
        logger.info(f"Finished training LSTM model. Final loss: {history.history['loss'][-1]:.6f}, "
                   f"Val loss: {val_loss:.6f}" if val_loss is not None else "")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the LSTM model.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        return self.model.predict(X).flatten()
    
    def save_weights(self, path: str):
        """
        Save model weights.
        
        Args:
            path: Path to save weights
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save weights
        self.model.save_weights(path)
        
        logger.info(f"Saved model weights to {path}")
    
    def load_weights(self, path: str):
        """
        Load model weights.
        
        Args:
            path: Path to load weights from
        """
        # Load weights
        self.model.load_weights(path)
        
        logger.info(f"Loaded model weights from {path}")
