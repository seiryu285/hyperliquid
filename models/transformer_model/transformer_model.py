"""
Transformer-based prediction model for time series forecasting.

This module implements a Transformer-based model for predicting price movements
using historical OHLCV data and technical indicators.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple, Optional
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, 
    MultiHeadAttention, GlobalAveragePooling1D
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from models.base_model import BaseModel
from core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class TransformerBlock(tf.keras.layers.Layer):
    """
    Transformer block consisting of Multi-Head Attention and Feed Forward Network.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout_rate: float = 0.1):
        """
        Initialize the transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed forward network dimension
            dropout_rate: Dropout rate
        """
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        
    def call(self, inputs, training=True):
        """
        Forward pass through the transformer block.
        
        Args:
            inputs: Input tensor
            training: Whether in training mode
            
        Returns:
            Output tensor
        """
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding layer for Transformer models.
    """
    
    def __init__(self, position: int, d_model: int):
        """
        Initialize the positional encoding layer.
        
        Args:
            position: Maximum sequence length
            d_model: Model dimension
        """
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angles(self, position, i, d_model):
        """
        Calculate angles for positional encoding.
        
        Args:
            position: Position vector
            i: Dimension vector
            d_model: Model dimension
            
        Returns:
            Angles
        """
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles
        
    def positional_encoding(self, position, d_model):
        """
        Calculate positional encoding.
        
        Args:
            position: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            Positional encoding tensor
        """
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sine to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cosine to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def call(self, inputs):
        """
        Forward pass through the positional encoding layer.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Output tensor with positional encoding added
        """
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]


class TransformerModel(BaseModel):
    """
    Transformer-based prediction model for time series forecasting.
    """
    
    def __init__(self, 
                input_dim: int,
                sequence_length: Optional[int] = None,
                prediction_horizon: Optional[int] = None,
                num_layers: Optional[int] = None,
                num_heads: Optional[int] = None,
                key_dim: Optional[int] = None,
                dropout: Optional[float] = None,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Transformer model.
        
        Args:
            input_dim: Number of input features
            sequence_length: Length of input sequences (lookback window)
            prediction_horizon: Number of time steps to predict into the future
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            key_dim: Dimension of keys in multi-head attention
            dropout: Dropout rate
            config: Additional model configuration
        """
        model_name = "transformer_model"
        self.sequence_length = sequence_length or settings.TRANSFORMER_SEQUENCE_LENGTH
        prediction_horizon = prediction_horizon or settings.LSTM_PREDICTION_HORIZON
        
        # Initialize base class
        super().__init__(model_name, input_dim, prediction_horizon, config)
        
        # Model specific parameters
        self.num_layers = num_layers or settings.TRANSFORMER_NUM_LAYERS
        self.num_heads = num_heads or settings.TRANSFORMER_NUM_HEADS
        self.key_dim = key_dim or settings.TRANSFORMER_KEY_DIM
        self.dropout = dropout or settings.TRANSFORMER_DROPOUT
        self.ff_dim = self.key_dim * 4  # Standard practice is to make FF dim 4x the key dim
        
        # Build the model
        self.build()
        
        logger.info(f"Initialized Transformer model with layers={self.num_layers}, "
                   f"heads={self.num_heads}, key_dim={self.key_dim}, "
                   f"dropout={self.dropout}")
                   
    def build(self) -> None:
        """Build the Transformer model architecture."""
        # Input layer
        inputs = Input(shape=(self.sequence_length, self.input_dim))
        
        # Add positional encoding
        x = PositionalEncoding(self.sequence_length, self.input_dim)(inputs)
        
        # Transformer blocks
        for _ in range(self.num_layers):
            x = TransformerBlock(
                self.input_dim, 
                self.num_heads, 
                self.ff_dim, 
                self.dropout
            )(x)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Output layers
        x = Dense(self.input_dim, activation="relu")(x)
        x = Dropout(self.dropout)(x)
        outputs = Dense(self.output_dim)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile the model
        optimizer = Adam(learning_rate=settings.LEARNING_RATE)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        logger.info(f"Built Transformer model with architecture: {model.summary()}")
        
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
        Train the Transformer model on the provided data.
        
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
        logger.info(f"Training Transformer model with {X_train.shape[0]} samples over {epochs} epochs")
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
        
        logger.info(f"Transformer model training completed. "
                   f"Final loss: {history.history['loss'][-1]:.4f}, "
                   f"Final val_loss: {history.history.get('val_loss', ['N/A'])[-1] if 'val_loss' in history.history else 'N/A'}")
        
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
        Save the Transformer model to disk.
        
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
        Load a trained Transformer model from disk.
        
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
            # Register custom layers for loading
            custom_objects = {
                'TransformerBlock': TransformerBlock,
                'PositionalEncoding': PositionalEncoding
            }
            
            # Load model
            self.model = load_model(model_path, custom_objects=custom_objects)
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
