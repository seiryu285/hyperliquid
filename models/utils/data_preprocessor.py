"""
Data preprocessing utilities for model training.

This module provides functions for preparing time series data for model training,
including sequence generation, scaling, train-test splitting, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from core.config import get_settings

# Get settings
settings = get_settings()


def create_sequences(data: np.ndarray, sequence_length: int, prediction_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for time series modeling.
    
    Args:
        data: Input data with shape (n_samples, n_features)
        sequence_length: Length of the sequences to create
        prediction_horizon: Number of steps to predict into the future
        
    Returns:
        Tuple of (X, y) where:
            X is a sequence of shape (n_samples-sequence_length-prediction_horizon+1, sequence_length, n_features)
            y is the target of shape (n_samples-sequence_length-prediction_horizon+1, prediction_horizon)
    """
    X, y = [], []
    
    # If data is 1D, reshape to 2D
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
    
    # Get the number of samples and features
    n_samples, n_features = data.shape
    
    # Create sequences
    for i in range(n_samples - sequence_length - prediction_horizon + 1):
        # Append sequence
        X.append(data[i:i+sequence_length])
        
        # Append target (use first feature if multiple features are available)
        y_values = data[i+sequence_length:i+sequence_length+prediction_horizon, 0]
        y.append(y_values)
    
    return np.array(X), np.array(y)


def train_val_test_split(X: np.ndarray, y: np.ndarray, 
                         val_size: float = None, 
                         test_size: float = None,
                         random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        X: Input sequences
        y: Target values
        val_size: Fraction of data to use for validation
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Use default values from settings if not provided
    val_size = val_size or settings.VALIDATION_SPLIT
    test_size = test_size or settings.TEST_SPLIT
    
    # Calculate the test size relative to the whole dataset
    if test_size > 0:
        # First, split off the test set
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # If val_size is 0, return without validation set
        if val_size == 0:
            return X_train_val, None, X_test, y_train_val, None, y_test
        
        # Split the remaining data into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=val_size/(1-test_size), 
            random_state=random_state, shuffle=False
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # If test_size is 0
    if val_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_size, random_state=random_state, shuffle=False
        )
        return X_train, X_val, None, y_train, y_val, None
    
    # If both val_size and test_size are 0
    return X, None, None, y, None, None


def scale_data(data: np.ndarray, scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None, 
               feature_range: Tuple[float, float] = (0, 1), return_scaler: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, Union[StandardScaler, MinMaxScaler]]]:
    """
    Scale the input data using MinMaxScaler or StandardScaler.
    
    Args:
        data: Input data to scale
        scaler: Pre-fitted scaler (if None, a new MinMaxScaler will be created)
        feature_range: Range for MinMaxScaler (ignored if scaler is provided)
        return_scaler: Whether to return the scaler along with the scaled data
        
    Returns:
        Scaled data if return_scaler is False, otherwise a tuple of (scaled_data, scaler)
    """
    # If data is 1D, reshape to 2D
    reshape_needed = False
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        reshape_needed = True
    
    # Create scaler if not provided
    if scaler is None:
        scaler = MinMaxScaler(feature_range=feature_range)
    
    # Fit and transform (or just transform if already fitted)
    if isinstance(scaler, (StandardScaler, MinMaxScaler)) and hasattr(scaler, 'n_features_in_'):
        # Already fitted
        scaled_data = scaler.transform(data)
    else:
        # Need to fit first
        scaled_data = scaler.fit_transform(data)
    
    # Reshape back to 1D if necessary
    if reshape_needed:
        scaled_data = scaled_data.ravel()
    
    if return_scaler:
        return scaled_data, scaler
    else:
        return scaled_data


def inverse_scale(data: np.ndarray, scaler: Union[StandardScaler, MinMaxScaler]) -> np.ndarray:
    """
    Inverse scale data using a fitted scaler.
    
    Args:
        data: Scaled data
        scaler: Fitted scaler used for the original scaling
        
    Returns:
        Data in original scale
    """
    # If data is 1D, reshape to 2D
    reshape_needed = False
    if len(data.shape) == 1:
        data = data.reshape(-1, 1)
        reshape_needed = True
    
    # Inverse transform
    original_data = scaler.inverse_transform(data)
    
    # Reshape back to 1D if necessary
    if reshape_needed:
        original_data = original_data.ravel()
    
    return original_data


def prepare_model_data(df: pd.DataFrame, 
                      target_column: str, 
                      feature_columns: List[str],
                      sequence_length: int,
                      prediction_horizon: int = 1,
                      val_size: float = None,
                      test_size: float = None,
                      use_standard_scaler: bool = False) -> Dict[str, Any]:
    """
    Prepare data for model training and evaluation.
    
    Args:
        df: DataFrame containing the data
        target_column: Column name for the prediction target
        feature_columns: List of column names to use as features
        sequence_length: Length of sequences
        prediction_horizon: Number of steps to predict
        val_size: Fraction of data for validation
        test_size: Fraction of data for testing
        use_standard_scaler: Whether to use StandardScaler instead of MinMaxScaler
        
    Returns:
        Dictionary containing:
            - X_train, X_val, X_test: Input sequences for each set
            - y_train, y_val, y_test: Target values for each set
            - scaler: The fitted scaler for future inverse transformations
            - original_target: The original target data before scaling
    """
    # Use default values from settings if not provided
    val_size = val_size or settings.VALIDATION_SPLIT
    test_size = test_size or settings.TEST_SPLIT
    
    # Extract features and target
    features = df[feature_columns].values
    target = df[target_column].values
    
    # Scale the features
    scaler_type = StandardScaler if use_standard_scaler else MinMaxScaler
    scaler = scaler_type()
    scaled_features = scale_data(features, scaler=scaler)
    
    # Scale the target (create a separate scaler for the target)
    target_scaler = MinMaxScaler()
    scaled_target = scale_data(target, scaler=target_scaler)
    
    # Create feature matrix by including the scaled target
    feature_matrix = np.column_stack([scaled_features, scaled_target])
    
    # Create sequences
    X, y = create_sequences(feature_matrix, sequence_length, prediction_horizon)
    
    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, val_size=val_size, test_size=test_size
    )
    
    # Return as dictionary
    return {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_scaler': scaler,
        'target_scaler': target_scaler,
        'original_target': target,
        'feature_columns': feature_columns,
        'target_column': target_column
    }
