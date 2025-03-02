"""
Main script for training price prediction models.

This script loads data, preprocesses it, and trains either LSTM or Transformer models
for price prediction. It can be run directly or imported and used as a module.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple, Optional
from datetime import datetime

from models.lstm_model.lstm_model import LSTMModel
from models.transformer_model.transformer_model import TransformerModel
from models.utils.data_preprocessor import prepare_model_data
from market_data.data_processor import DataProcessor
from core.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join('logs', 'model_training.log'))
    ]
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


def get_data(symbol: str, timeframe: str, 
             start_date: Optional[str] = None, 
             end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Get and preprocess data for model training.
    
    Args:
        symbol: Trading symbol
        timeframe: Timeframe (e.g., '1h', '4h', '1d')
        start_date: Start date for data retrieval
        end_date: End date for data retrieval
        
    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    # Initialize DataProcessor
    data_processor = DataProcessor(symbol=symbol)
    
    # Fetch OHLCV data
    df = data_processor.fetch_ohlcv_data(
        timeframe=timeframe,
        start_date=start_date,
        end_date=end_date
    )
    
    # Add technical indicators
    df = data_processor.add_all_indicators(df)
    
    # Drop rows with NaN values
    df = df.dropna()
    
    logger.info(f"Loaded {len(df)} rows of data for {symbol} on {timeframe} timeframe")
    return df


def train_lstm_model(data: pd.DataFrame, 
                    target_column: str = 'close',
                    feature_columns: Optional[List[str]] = None,
                    sequence_length: Optional[int] = None,
                    prediction_horizon: Optional[int] = None,
                    **kwargs) -> Tuple[LSTMModel, Dict[str, Any]]:
    """
    Train an LSTM model for price prediction.
    
    Args:
        data: DataFrame with features and target
        target_column: Column to predict
        feature_columns: Columns to use as features
        sequence_length: Number of time steps to look back
        prediction_horizon: Number of time steps to predict forward
        **kwargs: Additional parameters for the LSTM model
        
    Returns:
        Tuple of (trained model, preprocessed data dictionary)
    """
    # Set default parameters
    sequence_length = sequence_length or settings.LSTM_SEQUENCE_LENGTH
    prediction_horizon = prediction_horizon or settings.LSTM_PREDICTION_HORIZON
    
    # If feature columns not provided, use all except date
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != 'date' and col != target_column]
    
    # Prepare data for model training
    model_data = prepare_model_data(
        df=data,
        target_column=target_column,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Get input dimension (number of features)
    input_dim = model_data['X_train'].shape[2]
    
    # Initialize and build LSTM model
    lstm_model = LSTMModel(
        input_dim=input_dim,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        **kwargs
    )
    
    # Train the model
    lstm_model.train(
        X_train=model_data['X_train'],
        y_train=model_data['y_train'],
        X_val=model_data['X_val'],
        y_val=model_data['y_val']
    )
    
    # Evaluate the model
    if model_data['X_test'] is not None:
        metrics = lstm_model.evaluate(
            X_test=model_data['X_test'],
            y_test=model_data['y_test']
        )
        logger.info(f"LSTM Model Evaluation Metrics: {metrics}")
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(settings.MODEL_DIR, 'lstm_model', f"lstm_{timestamp}")
    lstm_model.save(save_path)
    
    return lstm_model, model_data


def train_transformer_model(data: pd.DataFrame, 
                           target_column: str = 'close',
                           feature_columns: Optional[List[str]] = None,
                           sequence_length: Optional[int] = None,
                           prediction_horizon: Optional[int] = None,
                           **kwargs) -> Tuple[TransformerModel, Dict[str, Any]]:
    """
    Train a Transformer model for price prediction.
    
    Args:
        data: DataFrame with features and target
        target_column: Column to predict
        feature_columns: Columns to use as features
        sequence_length: Number of time steps to look back
        prediction_horizon: Number of time steps to predict forward
        **kwargs: Additional parameters for the Transformer model
        
    Returns:
        Tuple of (trained model, preprocessed data dictionary)
    """
    # Set default parameters
    sequence_length = sequence_length or settings.TRANSFORMER_SEQUENCE_LENGTH
    prediction_horizon = prediction_horizon or settings.LSTM_PREDICTION_HORIZON
    
    # If feature columns not provided, use all except date
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != 'date' and col != target_column]
    
    # Prepare data for model training
    model_data = prepare_model_data(
        df=data,
        target_column=target_column,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon
    )
    
    # Get input dimension (number of features)
    input_dim = model_data['X_train'].shape[2]
    
    # Initialize and build Transformer model
    transformer_model = TransformerModel(
        input_dim=input_dim,
        sequence_length=sequence_length,
        prediction_horizon=prediction_horizon,
        **kwargs
    )
    
    # Train the model
    transformer_model.train(
        X_train=model_data['X_train'],
        y_train=model_data['y_train'],
        X_val=model_data['X_val'],
        y_val=model_data['y_val']
    )
    
    # Evaluate the model
    if model_data['X_test'] is not None:
        metrics = transformer_model.evaluate(
            X_test=model_data['X_test'],
            y_test=model_data['y_test']
        )
        logger.info(f"Transformer Model Evaluation Metrics: {metrics}")
    
    # Save the model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(settings.MODEL_DIR, 'transformer_model', f"transformer_{timestamp}")
    transformer_model.save(save_path)
    
    return transformer_model, model_data


def main():
    """Main function to train models from command line."""
    parser = argparse.ArgumentParser(description='Train price prediction models')
    parser.add_argument('--model', type=str, choices=['lstm', 'transformer'], 
                        default='lstm', help='Model type to train')
    parser.add_argument('--symbol', type=str, default='ETH', 
                        help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', 
                        help='Timeframe (e.g., 1h, 4h, 1d)')
    parser.add_argument('--target', type=str, default='close', 
                        help='Target column to predict')
    parser.add_argument('--start_date', type=str, 
                        help='Start date for data (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, 
                        help='End date for data (YYYY-MM-DD)')
    parser.add_argument('--sequence_length', type=int, 
                        help='Sequence length for lookback window')
    parser.add_argument('--prediction_horizon', type=int, 
                        help='Number of time steps to predict forward')
    
    args = parser.parse_args()
    
    # Get data
    data = get_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Train model
    if args.model == 'lstm':
        model, model_data = train_lstm_model(
            data=data,
            target_column=args.target,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon
        )
    else:
        model, model_data = train_transformer_model(
            data=data,
            target_column=args.target,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon
        )
    
    logger.info(f"Model training completed successfully")
    

if __name__ == "__main__":
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    main()
