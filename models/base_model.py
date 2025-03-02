"""
Base model class for prediction models.

This module defines the base class for all prediction models, ensuring a consistent
interface for training, evaluation, and prediction.
"""

import os
import logging
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union, Tuple, Optional
from datetime import datetime

from models.utils.metrics import calculate_all_metrics
from core.config import settings

# Configure logging
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for prediction models.
    
    All prediction models (LSTM, Transformer, etc.) should inherit from this class
    and implement the required methods.
    """
    
    def __init__(self, 
                model_name: str,
                input_dim: int,
                output_dim: int,
                config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model (used for saving/loading)
            input_dim: Input dimension (number of features)
            output_dim: Output dimension (prediction horizon)
            config: Dictionary with model configuration
        """
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.config = config or {}
        self.model_dir = os.path.join(settings.MODEL_DIR, model_name)
        self.is_trained = False
        self.model = None
        self.training_history = {}
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        logger.info(f"Initialized {model_name} with input_dim={input_dim}, output_dim={output_dim}")
    
    @abstractmethod
    def build(self) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             **kwargs) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary with training history/metrics
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions for the given input data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted values
        """
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            logger.warning("Model not trained yet. Cannot evaluate.")
            return {}
        
        # Generate predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        metrics = calculate_all_metrics(y_test, y_pred)
        
        logger.info(f"Evaluation metrics for {self.model_name}: {metrics}")
        return metrics
    
    def save(self, filepath: Optional[str] = None) -> str:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model (if None, use default path)
            
        Returns:
            Path where the model was saved
        """
        if not self.is_trained:
            logger.warning("Model not trained yet. Cannot save.")
            return ""
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(self.model_dir, f"{self.model_name}_{timestamp}")
        
        self._save_model(filepath)
        logger.info(f"Model saved to {filepath}")
        return filepath
    
    @abstractmethod
    def _save_model(self, filepath: str) -> None:
        """
        Save the model implementation to disk.
        
        Args:
            filepath: Path to save the model
        """
        pass
    
    @abstractmethod
    def load(self, filepath: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            True if loading was successful, False otherwise
        """
        pass
