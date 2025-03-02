"""
Inference module for making predictions with trained models.

This module provides functions for making predictions with trained LSTM or Transformer
models, as well as utilities for model retraining and evaluation.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple, Optional
from datetime import datetime, timedelta

from models.lstm_model.lstm_model import LSTMModel
from models.transformer_model.transformer_model import TransformerModel
from models.utils.data_preprocessor import prepare_model_data, inverse_scale
from market_data.data_processor import DataProcessor
from core.config import get_settings

# Configure logging
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


class ModelInference:
    """
    Class for making predictions with trained models.
    """
    
    MODEL_TYPES = {
        'lstm': LSTMModel,
        'transformer': TransformerModel
    }
    
    def __init__(self, 
                model_type: str,
                model_path: str,
                symbol: str = 'ETH',
                timeframe: str = '1h'):
        """
        Initialize the inference engine.
        
        Args:
            model_type: Type of model ('lstm' or 'transformer')
            model_path: Path to the saved model
            symbol: Trading symbol
            timeframe: Timeframe for data
        """
        self.model_type = model_type.lower()
        self.model_path = model_path
        self.symbol = symbol
        self.timeframe = timeframe
        
        # Validate model type
        if self.model_type not in self.MODEL_TYPES:
            raise ValueError(f"Model type must be one of {list(self.MODEL_TYPES.keys())}")
        
        # Initialize data processor
        self.data_processor = DataProcessor(symbol=symbol)
        
        # Model and data attributes
        self.model = None
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_columns = None
        self.target_column = None
        self.sequence_length = None
        self.prediction_horizon = None
        
        logger.info(f"Initialized model inference for {model_type} model at {model_path}")
    
    def load_model(self) -> bool:
        """
        Load the model and prepare for inference.
        
        Returns:
            True if successful, False otherwise
        """
        # Determine model class
        model_class = self.MODEL_TYPES[self.model_type]
        
        # Since we don't know the model's parameters yet, we'll load them
        # from the model file, so we create a temporary instance
        temp_model = model_class(input_dim=1, output_dim=1)
        
        # Try to load the model
        success = temp_model.load(self.model_path)
        if not success:
            logger.error(f"Failed to load model from {self.model_path}")
            return False
        
        # Update model attributes
        self.model = temp_model
        
        logger.info(f"Successfully loaded {self.model_type} model from {self.model_path}")
        return True
    
    def predict(self, data: pd.DataFrame, 
               target_column: str = 'close',
               feature_columns: Optional[List[str]] = None) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate predictions for the given data.
        
        Args:
            data: DataFrame with features
            target_column: Target column to predict
            feature_columns: Columns to use as features
            
        Returns:
            Tuple of (predictions array, DataFrame with predictions)
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return np.array([]), pd.DataFrame()
        
        # If feature columns not provided, use all except date
        if feature_columns is None:
            feature_columns = [col for col in data.columns if col != 'date' and col != target_column]
        
        # Store column information
        self.feature_columns = feature_columns
        self.target_column = target_column
        
        # Get sequence length from the model
        sequence_length = self.model.sequence_length
        prediction_horizon = self.model.output_dim
        
        # Store sequence info
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Prepare data for prediction
        model_data = prepare_model_data(
            df=data,
            target_column=target_column,
            feature_columns=feature_columns,
            sequence_length=sequence_length,
            prediction_horizon=prediction_horizon,
            test_size=0  # Use all data for prediction
        )
        
        # Store scalers for later inverse transformation
        self.feature_scaler = model_data['feature_scaler']
        self.target_scaler = model_data['target_scaler']
        
        # Make predictions
        X = model_data['X_train']  # Since test_size=0, all data is in X_train
        y_pred = self.model.predict(X)
        
        # Inverse transform predictions
        if self.target_scaler is not None:
            # Prepare a dummy array with the same shape as the feature matrix
            # to inverse transform just the target column
            dummy = np.zeros((y_pred.shape[0], 1))
            
            # For each prediction horizon step
            pred_horizons = []
            for i in range(y_pred.shape[1]):
                # Create the dummy array with the prediction at the right position
                dummy_pred = dummy.copy()
                dummy_pred[:, 0] = y_pred[:, i]
                
                # Inverse transform
                original_pred = inverse_scale(dummy_pred, self.target_scaler)
                pred_horizons.append(original_pred[:, 0])
            
            # Stack the predictions for each horizon
            original_y_pred = np.column_stack(pred_horizons)
        else:
            original_y_pred = y_pred
        
        # Create a DataFrame with the predictions
        if 'date' in data.columns:
            # Create date index for predictions (prediction starts after the sequence)
            dates = data['date'].values[sequence_length:]
            if len(dates) > len(original_y_pred):
                dates = dates[:len(original_y_pred)]
            elif len(dates) < len(original_y_pred):
                dates = np.append(dates, [pd.NaT] * (len(original_y_pred) - len(dates)))
            
            # Create prediction DataFrame
            pred_df = pd.DataFrame(index=range(len(original_y_pred)))
            pred_df['date'] = dates
            
            # Add predictions for each horizon
            for i in range(original_y_pred.shape[1]):
                horizon = i + 1
                pred_df[f'pred_{target_column}_h{horizon}'] = original_y_pred[:, i]
        else:
            # Create prediction DataFrame without dates
            pred_df = pd.DataFrame(index=range(len(original_y_pred)))
            for i in range(original_y_pred.shape[1]):
                horizon = i + 1
                pred_df[f'pred_{target_column}_h{horizon}'] = original_y_pred[:, i]
        
        logger.info(f"Generated predictions for {len(pred_df)} data points")
        return original_y_pred, pred_df
    
    def predict_latest(self, lookback_periods: int) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Generate predictions for the latest data.
        
        Args:
            lookback_periods: Number of periods to look back for data
            
        Returns:
            Tuple of (predictions array, DataFrame with predictions)
        """
        # Calculate lookback based on sequence length
        required_periods = self.sequence_length + lookback_periods
        
        # Fetch the latest data
        end_date = datetime.now()
        # Approximate start date (this depends on the timeframe)
        timeframe_to_hours = {
            '1m': 1/60,
            '5m': 5/60,
            '15m': 15/60,
            '30m': 30/60,
            '1h': 1,
            '4h': 4,
            '1d': 24
        }
        hours_per_period = timeframe_to_hours.get(self.timeframe, 1)
        start_date = end_date - timedelta(hours=required_periods * hours_per_period)
        
        # Fetch OHLCV data
        df = self.data_processor.fetch_ohlcv_data(
            timeframe=self.timeframe,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d %H:%M:%S')
        )
        
        # Add technical indicators
        df = self.data_processor.add_all_indicators(df)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Make predictions
        return self.predict(df)
    
    def evaluate_predictions(self, true_values: np.ndarray, 
                            predicted_values: np.ndarray) -> Dict[str, float]:
        """
        Evaluate predictions against true values.
        
        Args:
            true_values: Ground truth values
            predicted_values: Predicted values
            
        Returns:
            Dictionary with evaluation metrics
        """
        from models.utils.metrics import calculate_all_metrics
        
        # Reshape if needed
        if len(true_values.shape) == 1:
            true_values = true_values.reshape(-1, 1)
        
        if len(predicted_values.shape) == 1:
            predicted_values = predicted_values.reshape(-1, 1)
        
        # For multi-horizon predictions, evaluate each horizon separately
        metrics = {}
        
        # If multi-horizon, evaluate each horizon
        if predicted_values.shape[1] > 1:
            for i in range(predicted_values.shape[1]):
                # Get predictions for this horizon
                horizon_preds = predicted_values[:, i]
                
                # Get true values (same length as predictions)
                if i < true_values.shape[1]:
                    horizon_true = true_values[:, i]
                else:
                    # If we don't have true values for this horizon, skip
                    continue
                
                # Calculate metrics
                horizon_metrics = calculate_all_metrics(horizon_true, horizon_preds)
                
                # Add horizon prefix to metric names
                horizon_metrics = {f"h{i+1}_{k}": v for k, v in horizon_metrics.items()}
                
                # Add to overall metrics
                metrics.update(horizon_metrics)
        else:
            # Single horizon prediction
            metrics = calculate_all_metrics(true_values.ravel(), predicted_values.ravel())
        
        logger.info(f"Prediction evaluation metrics: {metrics}")
        return metrics
    
    def retrain_model(self, 
                     data: pd.DataFrame,
                     target_column: Optional[str] = None,
                     feature_columns: Optional[List[str]] = None) -> bool:
        """
        Retrain the model with new data.
        
        Args:
            data: New training data
            target_column: Target column to predict
            feature_columns: Columns to use as features
            
        Returns:
            True if retraining was successful, False otherwise
        """
        if self.model is None:
            logger.error("Model not loaded. Call load_model() first.")
            return False
        
        # Use stored columns if not provided
        target_column = target_column or self.target_column or 'close'
        if feature_columns is None:
            if self.feature_columns is not None:
                feature_columns = self.feature_columns
            else:
                feature_columns = [col for col in data.columns if col != 'date' and col != target_column]
        
        # Prepare data for retraining
        model_data = prepare_model_data(
            df=data,
            target_column=target_column,
            feature_columns=feature_columns,
            sequence_length=self.sequence_length,
            prediction_horizon=self.prediction_horizon
        )
        
        # Retrain the model
        try:
            self.model.train(
                X_train=model_data['X_train'],
                y_train=model_data['y_train'],
                X_val=model_data['X_val'],
                y_val=model_data['y_val']
            )
            
            # Update scalers
            self.feature_scaler = model_data['feature_scaler']
            self.target_scaler = model_data['target_scaler']
            
            # Save the retrained model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(
                settings.MODEL_DIR, 
                f"{self.model_type}_model", 
                f"{self.model_type}_retrained_{timestamp}"
            )
            self.model.save(save_path)
            
            logger.info(f"Model retrained successfully and saved to {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
            return False
